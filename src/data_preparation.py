"""
Data preparation utilities for the Credit Card Fraud Detection project.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """
    Load the credit card transactions dataset from CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)


def train_test_split_stratified(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split to preserve the fraud proportion.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_col : str, default "Class"
        Name of the target column.
    test_size : float, default 0.3
        Fraction of data to use for the test set.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
    y_test  : pd.Series
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    with_mean: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using sklearn's StandardScaler.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    with_mean : bool, default False
        Whether to center the data before scaling.
        (False is safer for highly sparse / PCA-like transforms.)

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : StandardScaler
        Fitted scaler, useful for inverse_transform or future data.
    """
    scaler = StandardScaler(with_mean=with_mean)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


__all__ = ["load_data", "train_test_split_stratified", "scale_features"]
