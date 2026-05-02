from typing import List, Tuple
import pandas as pd

class InputValidator:
    @staticmethod
    def validate_features(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[bool, str]:
        missing = [col for col in feature_cols + [target_col] if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"
        if df[feature_cols].isnull().any().any():
            return False, "Null values detected in feature columns"
        if (df[feature_cols] < 0).any().any():
            return False, "Negative spend values are not allowed"
        return True, "valid"

    @staticmethod
    def validate_prediction_inputs(rd: float, admin: float, marketing: float) -> Tuple[bool, str]:
        if any(v < 0 for v in [rd, admin, marketing]):
            return False, "All spend values must be non-negative"
        # Optional: add upper bounds based on historical data
        return True, "valid"