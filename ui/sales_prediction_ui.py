import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the enhanced prediction engine
from src.sales_prediction import SalesPredictionEngine

# Page configuration
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="",
    layout="wide"
)

# Cache the engine to avoid reloading on every interaction
@st.cache_resource
def load_engine():
    engine = SalesPredictionEngine(use_feature_engineering=True, auto_tune=False)
    engine.load_model()
    return engine

engine = load_engine()

# Sidebar navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Select Module",
    ["Single Prediction", "Batch Prediction", "Model Performance", "What-If Analysis"]
)

# Main header
st.title("Advanced Sales Prediction System")
st.markdown("---")

# ----------------------------------------------------------------------
# SINGLE PREDICTION (real-time) - Adjusted for actual data scale
# ----------------------------------------------------------------------
if mode == "Single Prediction":
    st.subheader("Real-Time Profit Predictor")
    st.markdown("Change any value below – the prediction updates instantly.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rd = st.number_input(
            "R&D Spend (₹)",
            min_value=0.0,
            max_value=500000.0,        # Increased to match CSV max (~350k)
            value=200000.0,            # Typical value from data
            step=10000.0,
            key="rd_single"
        )
    
    with col2:
        admin = st.number_input(
            "Administration Spend (₹)",
            min_value=0.0,
            max_value=300000.0,        # CSV max ~250k
            value=150000.0,            # Typical
            step=10000.0,
            key="admin_single"
        )
    
    with col3:
        marketing = st.number_input(
            "Marketing Spend (₹)",
            min_value=0.0,
            max_value=700000.0,        # CSV max ~600k
            value=350000.0,            # Typical
            step=10000.0,
            key="marketing_single"
        )
    
    # Real-time prediction
    try:
        summary = engine.generate_summary(rd, admin, marketing)
        
        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Investment", engine.format_currency(summary["total_investment"]))
        with m2:
            st.metric("Predicted Profit", engine.format_currency(summary["predicted_profit"]),
                      delta=f"{summary['roi_percent']:.1f}% ROI")
        with m3:
            if summary["predicted_profit"] > summary["total_investment"]:
                st.success("Break-even will be reached")
            else:
                st.warning("May not break even – consider adjusting spends")
        
        # Bar chart comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Investment", x=["R&D", "Admin", "Marketing"], y=[rd, admin, marketing]))
        fig.add_trace(go.Bar(name="Predicted Profit", x=["Profit"], y=[summary["predicted_profit"]]))
        fig.update_layout(title="Investment vs Predicted Profit", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown expander
        with st.expander("View Detailed Breakdown"):
            st.write(f"**R&D Spend:** {engine.format_currency(rd)}")
            st.write(f"**Administration Spend:** {engine.format_currency(admin)}")
            st.write(f"**Marketing Spend:** {engine.format_currency(marketing)}")
            st.write(f"**Total Investment:** {engine.format_currency(summary['total_investment'])}")
            st.write(f"**Predicted Profit:** {engine.format_currency(summary['predicted_profit'])}")
            st.write(f"**Return on Investment (ROI):** {summary['roi_percent']:.2f}%")
            net = summary["predicted_profit"] - summary["total_investment"]
            if net >= 0:
                st.success(f"**Net Gain:** {engine.format_currency(net)}")
            else:
                st.error(f"**Net Loss:** {engine.format_currency(abs(net))}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ----------------------------------------------------------------------
# BATCH PREDICTION (CSV upload) - unchanged
# ----------------------------------------------------------------------
elif mode == "Batch Prediction":
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV with columns: `R&D_Spend`, `Administration`, `Marketing_Spend`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df_input.head())
        
        if st.button("Run Batch Prediction", type="primary"):
            try:
                # Save temporarily
                temp_path = Path("/tmp/uploaded_batch.csv")
                temp_path.parent.mkdir(exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                result_df = engine.batch_predict(temp_path)
                st.success("Predictions completed successfully")
                st.dataframe(result_df)
                
                # Download button
                csv_output = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_output,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # Visualization
                fig = px.bar(
                    result_df,
                    x=result_df.index,
                    y="Predicted_Profit",
                    title="Predicted Profit per Row",
                    labels={"index": "Row Number", "Predicted_Profit": "Profit (₹)"}
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Batch prediction error: {e}")

# ----------------------------------------------------------------------
# MODEL PERFORMANCE (metrics and retraining) - unchanged
# ----------------------------------------------------------------------
elif mode == "Model Performance":
    st.subheader("Model Performance Metrics")
    st.markdown("These metrics reflect the accuracy of the latest trained model.")
    
    metrics = engine.get_model_metrics()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
        with col2:
            st.metric("Mean Absolute Error", engine.format_currency(metrics.get('mae', 0)))
        with col3:
            st.metric("Root Mean Squared Error", engine.format_currency(metrics.get('rmse', 0)))
        with col4:
            st.metric("Cross-Validation R²", f"{metrics.get('cv_score', 0):.4f}")
        
        st.markdown("---")
        st.markdown("""
        **Interpretation:**
        - **R² Score** (closer to 1.0): Proportion of profit variance explained by the model.
        - **MAE / RMSE**: Average prediction errors in Rupees (lower is better).
        - **Cross-Validation R²**: Consistency across different data splits.
        """)
    else:
        st.warning("No trained model found. Please train the model first.")
    
    if st.button("Retrain Model Now", type="primary"):
        with st.spinner("Retraining model with current data... This may take a few seconds."):
            new_metrics = engine.train_model(save_versioned=True)
            st.success(f"Retraining complete! New R² score: {new_metrics['r2_score']:.4f}")
            st.rerun()

# ----------------------------------------------------------------------
# WHAT-IF ANALYSIS (sensitivity heatmap) - Adjusted ranges
# ----------------------------------------------------------------------
elif mode == "What-If Analysis":
    st.subheader("What-If Scenario Analysis")
    st.markdown("Simultaneously vary R&D and Marketing spends to see profit sensitivity.")
    
    col_fixed, col_rd, col_mkt = st.columns([1, 2, 2])
    with col_fixed:
        admin_fixed = st.number_input(
            "Fixed Administration Spend (₹)",
            min_value=0,
            max_value=300000,          # Increased from 150k
            value=150000,              # Typical
            step=10000,
            key="admin_fixed"
        )
    
    with col_rd:
        rd_min = st.number_input("R&D Min (₹)", 0, 500000, 100000, 10000)
        rd_max = st.number_input("R&D Max (₹)", 0, 500000, 400000, 10000)
    with col_mkt:
        mkt_min = st.number_input("Marketing Min (₹)", 0, 700000, 150000, 10000)
        mkt_max = st.number_input("Marketing Max (₹)", 0, 700000, 600000, 10000)
    
    if st.button("Generate Heatmap"):
        if rd_min >= rd_max or mkt_min >= mkt_max:
            st.error("Min must be less than Max.")
        else:
            rd_range = list(range(rd_min, rd_max + 1, max(1, (rd_max - rd_min)//10)))
            mkt_range = list(range(mkt_min, mkt_max + 1, max(1, (mkt_max - mkt_min)//10)))
            profit_matrix = []
            for rd_val in rd_range:
                row = []
                for mkt_val in mkt_range:
                    try:
                        profit = engine.predict(rd_val, admin_fixed, mkt_val)
                        row.append(profit)
                    except:
                        row.append(0)
                profit_matrix.append(row)
            
            fig = px.imshow(
                profit_matrix,
                x=mkt_range,
                y=rd_range,
                labels=dict(x="Marketing Spend (₹)", y="R&D Spend (₹)", color="Predicted Profit (₹)"),
                title=f"Profit Heatmap (Admin fixed at {engine.format_currency(admin_fixed)})",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig)