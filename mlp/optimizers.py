"""Optimiseurs (descente de gradient).

- SGD : descente de gradient stochastique, avec momentum optionnel.
- Adam : optimiseur adaptatif (bonus du sujet).

Tous opèrent en place sur le dictionnaire de paramètres {W1, b1, W2, ...} à
partir des gradients {dW1, db1, ...}.
"""

import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params:
            g = grads["d" + key]
            if self.momentum:
                v = self.momentum * self.velocity.get(key, 0.0) - self.lr * g
                self.velocity[key] = v
                params[key] += v
            else:
                params[key] -= self.lr * g


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            g = grads["d" + key]
            self.m[key] = self.beta1 * self.m.get(key, 0.0) + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v.get(key, 0.0) + (1 - self.beta2) * (g * g)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "momentum":
        return SGD(lr=lr, momentum=0.9)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(f"Optimiseur inconnu : {name} (sgd|momentum|adam)")
