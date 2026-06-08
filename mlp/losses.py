"""Fonctions de coût.

- categorical_cross_entropy : coût d'entraînement (softmax + one-hot).
- binary_cross_entropy : fonction d'erreur demandée par le sujet pour la phase
  de prédiction, E = -(1/N) * somme[ y*log(p) + (1-y)*log(1-p) ].
"""

import numpy as np

EPS = 1e-12  # évite log(0)


def categorical_cross_entropy(Y, A):
    """Y, A de forme (n_classes, m). A est la sortie softmax."""
    m = Y.shape[1]
    return float(-np.sum(Y * np.log(A + EPS)) / m)


def binary_cross_entropy(y_true, p):
    """y_true (m,) dans {0,1} ; p (m,) = probabilité de la classe positive."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


# Noms acceptés par --loss. Pour une sortie softmax à 2 classes, l'entropie
# croisée catégorielle et binaire sont équivalentes ; on entraîne avec la CCE.
LOSSES = {
    "categoricalCrossentropy": categorical_cross_entropy,
    "binaryCrossentropy": categorical_cross_entropy,
}
