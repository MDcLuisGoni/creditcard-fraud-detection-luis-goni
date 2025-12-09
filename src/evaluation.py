"""
Evaluation utilities for the Credit Card Fraud Detection project.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute main fraud-detection metrics for a single model.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales (0 = no fraude, 1 = fraude).
    y_proba : array-like
        Probabilidades predichas para la clase 1 (fraude).
    threshold : float, default=0.5
        Umbral para convertir probabilidad en clase 0/1.

    Returns
    -------
    dict
        Diccionario con:
        - roc_auc
        - pr_auc
        - precision_fraud
        - recall_fraud
        - f1_fraud
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_fraud": precision,
        "recall_fraud": recall,
        "f1_fraud": f1,
    }


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate multiple models and return a metrics table.

    Parameters
    ----------
    models : dict
        Diccionario {nombre_modelo: modelo_entrenado}.
        Los modelos deben tener `predict_proba`.
    X_test, y_test
        Datos de test.
    threshold : float
        Umbral para convertir probabilidad en clase 0/1.

    Returns
    -------
    pd.DataFrame
        Tabla con una fila por modelo y columnas:
        ['roc_auc', 'pr_auc', 'precision_fraud', 'recall_fraud', 'f1_fraud']
    """
    rows = []

    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba, threshold)
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    results = pd.DataFrame(rows).set_index("model").sort_values(
        by="pr_auc", ascending=False
    )
    return results


def plot_precision_recall_curves(
    models: Dict[str, object],
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray,
    ax: plt.Axes | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Precision–Recall curves for multiple models.

    Devuelve figura y ejes por si querés guardarla en `figures/pr_curves.png`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve for Fraud Detection")
    ax.legend()
    ax.grid(True)

    return fig, ax


__all__ = ["compute_metrics", "evaluate_models", "plot_precision_recall_curves"]
