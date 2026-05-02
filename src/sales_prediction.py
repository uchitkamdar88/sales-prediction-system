from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Union, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Try importing XGBoost (optional but recommended)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Falling back to GradientBoosting.")

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("SalesPrediction")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # File handler
    fh = logging.FileHandler(LOG_DIR / "sales_prediction.log")
    fh.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Enhanced Sales Prediction Engine
# ---------------------------------------------------------

class SalesPredictionEngine:
    """
    Advanced ML engine for sales/profit prediction.
    
    Features:
    - Polynomial feature engineering (degree 2) to capture interactions
    - XGBoost / GradientBoosting / LinearRegression (auto-selected)
    - Model versioning and automatic loading of best model
    - Cross-validation and hyperparameter tuning
    - Batch prediction from CSV
    - Full logging and error handling
    """
    
    FEATURES = ["R&D_Spend", "Administration", "Marketing_Spend"]
    TARGET = "Profit"
    
    def __init__(self, use_feature_engineering: bool = True, auto_tune: bool = False):
        """
        Initialize the prediction engine.
        
        Args:
            use_feature_engineering: If True, adds polynomial/interaction features.
            auto_tune: If True, performs hyperparameter tuning (slower but better).
        """
        self.model = None
        self.poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False) if use_feature_engineering else None
        self.feature_names = None
        self.use_feature_engineering = use_feature_engineering
        self.auto_tune = auto_tune
        self.metrics = {}
        
    # -----------------------------------------------------
    # Currency formatter
    # -----------------------------------------------------
    
    @staticmethod
    def format_currency(amount: float) -> str:
        """Format amount as Indian Rupees."""
        amount = int(round(amount))
        return f"₹{amount:,}"
    
    # -----------------------------------------------------
    # Dataset loading and validation
    # -----------------------------------------------------
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate dataset."""
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
        
        data = pd.read_csv(DATA_PATH)
        required_columns = self.FEATURES + [self.TARGET]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")
        
        # Remove rows with any missing values
        initial_len = len(data)
        data = data.dropna()
        if len(data) < initial_len:
            logger.warning(f"Dropped {initial_len - len(data)} rows with missing values.")
        
        # Check for negative spends
        negative_in_features = (data[self.FEATURES] < 0).any().any()
        if negative_in_features:
            logger.warning("Negative spend values detected. They will be used but ROI may be nonsense.")
        
        logger.info(f"Loaded {len(data)} rows from {DATA_PATH}")
        return data
    
    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------
    
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply polynomial feature expansion."""
        if not self.use_feature_engineering:
            return X
        
        # Fit or transform
        if self.poly is None:
            raise ValueError("PolynomialFeatures not initialized.")
        
        # Store original feature names for later reference
        if self.feature_names is None:
            # First time – fit
            X_poly = self.poly.fit_transform(X)
            self.feature_names = self.poly.get_feature_names_out(X.columns)
        else:
            # Transform only
            X_poly = self.poly.transform(X)
        
        return pd.DataFrame(X_poly, columns=self.feature_names, index=X.index)
    
    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------
    
    def _select_model(self):
        """Choose the best available model based on config and availability."""
        if XGB_AVAILABLE:
            model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 verbosity=0)
            logger.info("Using XGBoost regressor.")
        else:
            model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                              learning_rate=0.05, random_state=42)
            logger.info("Using GradientBoosting regressor (XGBoost not installed).")
        
        if self.auto_tune:
            logger.info("Auto-tuning enabled. Running GridSearchCV (may take time).")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }
            if XGB_AVAILABLE:
                grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            else:
                grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            # We'll fit later, return the grid object
            return grid
        else:
            return model
    
    # -----------------------------------------------------
    # Model training
    # -----------------------------------------------------
    
    def train_model(self, save_versioned: bool = True) -> Dict[str, float]:
        """
        Train the model and save it to disk.
        
        Args:
            save_versioned: If True, also save a timestamped copy.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Starting model training...")
        data = self.load_data()
        X_raw = data[self.FEATURES]
        y = data[self.TARGET]
        
        # Feature engineering
        X = self._create_features(X_raw)
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        # Model selection
        model_or_grid = self._select_model()
        
        if self.auto_tune and isinstance(model_or_grid, GridSearchCV):
            model_or_grid.fit(X_train, y_train)
            self.model = model_or_grid.best_estimator_
            logger.info(f"Best parameters: {model_or_grid.best_params_}")
        else:
            model_or_grid.fit(X_train, y_train)
            self.model = model_or_grid
        
        # Predictions and metrics
        y_pred = self.model.predict(X_test)
        self.metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "cv_score": cross_val_score(self.model, X, y, cv=5, scoring='r2').mean()
        }
        logger.info(f"Training metrics: {self.metrics}")
        
        # Save model
        model_artifact = {
            "model": self.model,
            "poly": self.poly,
            "feature_names": self.feature_names,
            "use_feature_engineering": self.use_feature_engineering,
            "metrics": self.metrics,
            "training_date": datetime.now().isoformat()
        }
        model_path = MODEL_DIR / "trained_model.pkl"
        joblib.dump(model_artifact, model_path)
        logger.info(f"Model saved to {model_path}")
        
        if save_versioned:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_path = MODEL_DIR / f"model_{timestamp}.pkl"
            joblib.dump(model_artifact, versioned_path)
            logger.info(f"Versioned model saved to {versioned_path}")
        
        return self.metrics
    
    # -----------------------------------------------------
    # Model loading
    # -----------------------------------------------------
    
    def load_model(self) -> object:
        """Load the latest trained model. If none exists, train automatically."""
        model_path = MODEL_DIR / "trained_model.pkl"
        if model_path.exists():
            artifact = joblib.load(model_path)
            self.model = artifact["model"]
            self.poly = artifact.get("poly")
            self.feature_names = artifact.get("feature_names")
            self.use_feature_engineering = artifact.get("use_feature_engineering", False)
            self.metrics = artifact.get("metrics", {})
            logger.info(f"Model loaded from {model_path} (trained on {artifact.get('training_date', 'unknown')})")
            return self.model
        else:
            logger.warning("No saved model found. Training a new model.")
            self.train_model()
            return self.model
    
    # -----------------------------------------------------
    # Input validation
    # -----------------------------------------------------
    
    @staticmethod
    def validate_inputs(rd_spend: float, admin_spend: float, marketing_spend: float) -> None:
        """Raise ValueError if any input is negative."""
        if any(v < 0 for v in [rd_spend, admin_spend, marketing_spend]):
            raise ValueError("All spend values must be non-negative.")
    
    # -----------------------------------------------------
    # Single prediction
    # -----------------------------------------------------
    
    def predict(self, rd_spend: float, admin_spend: float, marketing_spend: float) -> float:
        """
        Predict profit for given spends.
        
        Returns:
            Predicted profit as float.
        """
        self.validate_inputs(rd_spend, admin_spend, marketing_spend)
        if self.model is None:
            self.load_model()
        
        # Create raw input DataFrame
        input_raw = pd.DataFrame([[rd_spend, admin_spend, marketing_spend]],
                                 columns=self.FEATURES)
        
        # Apply feature engineering if used
        if self.use_feature_engineering and self.poly is not None:
            input_poly = self.poly.transform(input_raw)
            # Ensure columns match training (there could be extra zeros)
            if self.feature_names is not None:
                input_df = pd.DataFrame(input_poly, columns=self.feature_names)
            else:
                input_df = pd.DataFrame(input_poly)
        else:
            input_df = input_raw
        
        prediction = self.model.predict(input_df)[0]
        logger.debug(f"Prediction for ({rd_spend}, {admin_spend}, {marketing_spend}) = {prediction}")
        return float(prediction)
    
    # -----------------------------------------------------
    # Business summary
    # -----------------------------------------------------
    
    def generate_summary(self, rd_spend: float, admin_spend: float, marketing_spend: float) -> Dict[str, float]:
        """
        Return a business-friendly summary including ROI and total investment.
        """
        predicted_profit = self.predict(rd_spend, admin_spend, marketing_spend)
        total_investment = rd_spend + admin_spend + marketing_spend
        roi = ((predicted_profit - total_investment) / total_investment) * 100 if total_investment > 0 else 0.0
        
        return {
            "rd_spend": rd_spend,
            "admin_spend": admin_spend,
            "marketing_spend": marketing_spend,
            "total_investment": total_investment,
            "predicted_profit": predicted_profit,
            "roi_percent": roi
        }
    
    # -----------------------------------------------------
    # Batch prediction (new feature)
    # -----------------------------------------------------
    
    def batch_predict(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """
        Predict profits for multiple rows in a CSV file.
        
        The CSV must contain columns: R&D_Spend, Administration, Marketing_Spend.
        
        Returns:
            Original DataFrame with added 'Predicted_Profit' and 'ROI_%' columns.
        """
        df = pd.read_csv(csv_path)
        missing = [col for col in self.FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        predictions = []
        for _, row in df.iterrows():
            pred = self.predict(row["R&D_Spend"], row["Administration"], row["Marketing_Spend"])
            predictions.append(pred)
        
        df["Predicted_Profit"] = predictions
        total_invest = df[self.FEATURES].sum(axis=1)
        df["ROI_%"] = ((df["Predicted_Profit"] - total_invest) / total_invest) * 100
        df["ROI_%"] = df["ROI_%"].fillna(0.0)
        logger.info(f"Batch prediction completed for {len(df)} rows.")
        return df
    
    # -----------------------------------------------------
    # Model metrics access
    # -----------------------------------------------------
    
    def get_model_metrics(self) -> Dict[str, float]:
        """Return the latest training metrics."""
        if not self.metrics:
            self.load_model()
        return self.metrics
    
    # -----------------------------------------------------
    # Retrain with new data
    # -----------------------------------------------------
    
    def retrain_with_new_data(self, new_data_path: Union[str, Path], merge: bool = True) -> Dict[str, float]:
        """
        Append new data to existing dataset and retrain model.
        
        Args:
            new_data_path: Path to CSV with same columns.
            merge: If True, combines with existing data; if False, replaces.
        """
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Original data not found at {DATA_PATH}")
        
        existing = pd.read_csv(DATA_PATH)
        new_data = pd.read_csv(new_data_path)
        
        required = self.FEATURES + [self.TARGET]
        for col in required:
            if col not in new_data.columns:
                raise ValueError(f"New data missing column: {col}")
        
        if merge:
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data
        
        # Save combined data
        combined.to_csv(DATA_PATH, index=False)
        logger.info(f"Updated dataset saved to {DATA_PATH}. Total rows: {len(combined)}")
        
        # Retrain
        return self.train_model(save_versioned=True)


# ---------------------------------------------------------
# Standalone execution (CLI)
# ---------------------------------------------------------

if __name__ == "__main__":
    engine = SalesPredictionEngine(use_feature_engineering=True, auto_tune=False)
    metrics = engine.train_model()
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)
    print(f"R² Score      : {metrics['r2_score']:.4f}")
    print(f"MAE           : {engine.format_currency(metrics['mae'])}")
    print(f"RMSE          : {engine.format_currency(metrics['rmse'])}")
    print(f"5-Fold CV R²  : {metrics['cv_score']:.4f}")
    print("="*50)
    
    # Quick demonstration prediction
    print("\nDemo Prediction:")
    test_rd, test_admin, test_marketing = 60000, 35000, 45000
    summary = engine.generate_summary(test_rd, test_admin, test_marketing)
    print(f"Inputs: R&D={test_rd}, Admin={test_admin}, Marketing={test_marketing}")
    print(f"Predicted Profit: {engine.format_currency(summary['predicted_profit'])}")
    print(f"ROI: {summary['roi_percent']:.2f}%")