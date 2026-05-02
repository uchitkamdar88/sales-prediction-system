import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import config
from src.logger import setup_logger
from src.data_validator import InputValidator
from src.feature_engineering import FeatureEngineer

logger = setup_logger("model_trainer")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_engineer = FeatureEngineer(degree=2, include_interaction_only=True)
        self.model_type = config["model"]["type"]
        self.params = config["model"]["hyperparameters"]

    def load_and_prepare_data(self):
        data_path = Path(config["data"]["path"])
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        feature_cols = config["data"]["feature_columns"]
        target_col = config["data"]["target_column"]

        valid, msg = InputValidator.validate_features(df, feature_cols, target_col)
        if not valid:
            raise ValueError(f"Data validation failed: {msg}")

        X = df[feature_cols]
        y = df[target_col]

        # Feature engineering (polynomial + interaction)
        X_engineered = self.feature_engineer.fit_transform(X)

        return X_engineered, y

    def _create_model(self):
        if self.model_type == "xgboost":
            return XGBRegressor(**self.params)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(**self.params)
        else:
            return LinearRegression()

    def train(self, save_versioned: bool = True):
        X, y = self.load_and_prepare_data()
        test_size = config["training"]["test_size"]
        random_state = config["training"]["random_state"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        metrics = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "cv_score": cross_val_score(self.model, X, y, cv=config["training"]["cv_folds"]).mean()
        }

        logger.info(f"Training completed. Metrics: {metrics}")

        # Save model
        model_file = MODEL_DIR / "trained_model.pkl"
        joblib.dump({
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "metrics": metrics,
            "feature_names": X.columns.tolist()
        }, model_file)

        if save_versioned:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_file = MODEL_DIR / f"model_{timestamp}.pkl"
            joblib.dump({
                "model": self.model,
                "feature_engineer": self.feature_engineer,
                "metrics": metrics,
                "feature_names": X.columns.tolist()
            }, versioned_file)

        return metrics

    def load_latest_model(self):
        model_path = MODEL_DIR / "trained_model.pkl"
        if not model_path.exists():
            logger.info("No existing model found. Training new model.")
            self.train()
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_engineer = artifact["feature_engineer"]
        self.feature_names = artifact["feature_names"]
        logger.info("Model loaded successfully.")
        return self.model