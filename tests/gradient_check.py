#!/usr/bin/env python3
"""Vérification numérique du gradient (gradient checking).

Compare les gradients de la rétropropagation à une estimation par différences
finies. Un écart relatif < 1e-6 prouve que la backprop est mathématiquement
correcte. Utile à montrer en soutenance.

    python tests/gradient_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp.losses import categorical_cross_entropy  # noqa: E402
from mlp.network import MLP  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    n_in, m = 5, 8
    X = rng.standard_normal((n_in, m))
    y = rng.integers(0, 2, size=m)
    Y = np.eye(2)[y].T

    net = MLP([n_in, 6, 4, 2], hidden_activation="relu", seed=1)
    _, cache = net.forward(X)
    grads = net.backward(Y, cache)

    eps = 1e-6
    max_rel = 0.0
    for key in net.params:
        W = net.params[key]
        grad_analytic = grads["d" + key]
        for _ in range(40):  # échantillon de coordonnées
            i = tuple(rng.integers(0, s) for s in W.shape)
            old = W[i]
            W[i] = old + eps
            plus = categorical_cross_entropy(Y, net.forward(X)[0])
            W[i] = old - eps
            minus = categorical_cross_entropy(Y, net.forward(X)[0])
            W[i] = old
            grad_num = (plus - minus) / (2 * eps)
            denom = max(1e-8, abs(grad_num) + abs(grad_analytic[i]))
            max_rel = max(max_rel, abs(grad_num - grad_analytic[i]) / denom)

    print(f"écart relatif max backprop vs numérique : {max_rel:.2e}")
    assert max_rel < 1e-6, "backprop incorrecte !"
    print("OK : la rétropropagation est correcte.")


if __name__ == "__main__":
    main()
