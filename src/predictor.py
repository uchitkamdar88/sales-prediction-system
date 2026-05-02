import numpy as np
import pandas as pd
from src.model_trainer import ModelTrainer
from src.data_validator import InputValidator
from src.logger import setup_logger

logger = setup_logger("predictor")

class SalesPredictor:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.trainer.load_latest_model()
        self.model = self.trainer.model
        self.feature_engineer = self.trainer.feature_engineer

    def predict(self, rd_spend: float, admin_spend: float, marketing_spend: float) -> dict:
        valid, msg = InputValidator.validate_prediction_inputs(rd_spend, admin_spend, marketing_spend)
        if not valid:
            raise ValueError(msg)

        # Create base DataFrame
        input_df = pd.DataFrame([[
            rd_spend, admin_spend, marketing_spend
        ]], columns=config["data"]["feature_columns"])

        # Apply feature engineering (polynomial + interactions)
        input_engineered = self.feature_engineer.transform(input_df)

        # Ensure columns match training
        expected_cols = self.trainer.feature_names
        for col in expected_cols:
            if col not in input_engineered.columns:
                input_engineered[col] = 0
        input_engineered = input_engineered[expected_cols]

        prediction = self.model.predict(input_engineered)[0]

        total_investment = rd_spend + admin_spend + marketing_spend
        roi = (prediction - total_investment) / total_investment * 100 if total_investment > 0 else 0

        return {
            "predicted_profit": float(prediction),
            "total_investment": float(total_investment),
            "roi_percent": float(roi),
            "break_even": total_investment < prediction
        }

    def batch_predict(self, csv_file_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_file_path)
        required = config["data"]["feature_columns"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        predictions = []
        for _, row in df.iterrows():
            res = self.predict(row["R&D_Spend"], row["Administration"], row["Marketing_Spend"])
            predictions.append(res["predicted_profit"])

        df["Predicted_Profit"] = predictions
        df["ROI_%"] = [(p - (r+a+m)) / (r+a+m) * 100 if (r+a+m)>0 else 0
                       for p, r, a, m in zip(predictions, df["R&D_Spend"], df["Administration"], df["Marketing_Spend"])]
        return df