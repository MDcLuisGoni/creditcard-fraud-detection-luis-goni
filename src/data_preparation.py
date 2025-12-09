"""
Data preparation utilities for the Credit Card Fraud Detection project.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load the credit card transactions dataset."""
    return pd.read_csv(path)


def train_test_split_stratified(df: pd.DataFrame, target_col: str = "Class", test_size: float = 0.3, random_state: int = 42):
    """
    Stratified train/test split to mantener la proporción de fraude.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(X_train, X_test, with_mean: bool = False):
    """
    StandardScaler: en este dataset suele usarse with_mean=False por PCA.
    """
    scaler = StandardScaler(with_mean=with_mean)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
