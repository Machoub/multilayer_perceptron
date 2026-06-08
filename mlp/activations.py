"""Fonctions d'activation et leurs dérivées.

Convention : toutes les fonctions reçoivent la pré-activation Z (et non
l'activation A), pour que la dérivée soit calculée correctement quel que soit
l'endroit où elle est appelée.
"""

import numpy as np


def relu(Z):
    return np.maximum(0.0, Z)


def relu_deriv(Z):
    # dérivée par rapport à Z (1 si Z > 0, sinon 0)
    return (Z > 0).astype(Z.dtype)


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_deriv(Z):
    s = sigmoid(Z)
    return s * (1.0 - s)


def softmax(Z):
    # stabilité numérique : on soustrait le max par colonne avant l'exponentielle
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


# Registre (fonction, dérivée) accessible par nom — utilisé par le réseau pour
# rester générique (aucune activation codée en dur dans la boucle).
ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
}
