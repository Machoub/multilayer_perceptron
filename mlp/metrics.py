"""Métriques d'évaluation (classe positive = 1 = malin)."""

import numpy as np


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


def confusion(y_pred, y_true):
    """Renvoie (TP, TN, FP, FN) pour la classe positive 1."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def precision_recall_f1(y_pred, y_true):
    tp, tn, fp, fn = confusion(y_pred, y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
