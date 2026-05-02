import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    def __init__(self, degree: int = 2, include_interaction_only: bool = False):
        self.degree = degree
        self.include_interaction_only = include_interaction_only
        self.poly = PolynomialFeatures(degree=degree, interaction_only=include_interaction_only, include_bias=False)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        poly_features = self.poly.fit_transform(X)
        feature_names = self.poly.get_feature_names_out(X.columns)
        return pd.DataFrame(poly_features, columns=feature_names, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        poly_features = self.poly.transform(X)
        feature_names = self.poly.get_feature_names_out(X.columns)
        return pd.DataFrame(poly_features, columns=feature_names, index=X.index)