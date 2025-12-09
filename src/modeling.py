"""
Model definitions for the Credit Card Fraud Detection project.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_baseline_models(random_state: int = 42) -> Dict[str, object]:
    """
    Create the baseline models used in the notebooks.

    Returns
    -------
    Dict[str, object]
        Dictionary with untrained sklearn estimators.
    """
    log_reg = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
        class_weight=None,  # baseline sin balanceo explícito
    )

    linear_svc = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
    )

    # Calibramos para obtener probabilidades (para ROC/PR-AUC)
    svm_calibrated = CalibratedClassifierCV(
        base_estimator=linear_svc,
        method="sigmoid",
        cv=3,
    )

    models = {
        "Logistic Regression (Baseline)": log_reg,
        "Random Forest": rf,
        "LinearSVC Calibrated": svm_calibrated,
    }

    return models


def fit_models(
    models: Dict[str, object],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> Dict[str, object]:
    """
    Fit all models in the dictionary.

    Parameters
    ----------
    models : dict
        Diccionario de modelos devuelto por `get_baseline_models`.
    X_train, y_train
        Datos de entrenamiento.

    Returns
    -------
    Dict[str, object]
        Diccionario con los modelos ya entrenados.
    """
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


__all__ = ["get_baseline_models", "fit_models"]
