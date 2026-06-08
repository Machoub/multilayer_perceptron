"""Le perceptron multicouche, générique sur le nombre et la taille des couches.

Convention matricielle (colonnes = exemples) :
  - X      : (n_features, m)
  - W[l]   : (n_l, n_{l-1})
  - b[l]   : (n_l, 1)
  - Z[l] = W[l] @ A[l-1] + b[l]      (A[0] = X)
  - A[l] = activation_cachée(Z[l]) pour l < L, softmax(Z[L]) en sortie.

Le réseau est construit à partir d'une LISTE de tailles de couches
(layer_sizes), donc la profondeur et la largeur sont des données, pas du code
codé en dur. C'est ce qui rend l'architecture modulaire.
"""

import numpy as np

from .activations import ACTIVATIONS, softmax


class MLP:
    def __init__(self, layer_sizes, hidden_activation="relu", seed=None):
        self.layer_sizes = [int(s) for s in layer_sizes]   # [n_in, h1, ..., n_out]
        self.hidden_activation = hidden_activation
        self.L = len(self.layer_sizes) - 1                 # nombre de couches de poids
        self.params = {}
        self._init_parameters(seed)

    def _init_parameters(self, seed):
        """Initialisation He (relu) ou Xavier (sigmoid) ; biais à zéro."""
        rng = np.random.default_rng(seed)
        for l in range(1, self.L + 1):
            fan_in = self.layer_sizes[l - 1]
            fan_out = self.layer_sizes[l]
            if l < self.L and self.hidden_activation == "relu":
                scale = np.sqrt(2.0 / fan_in)              # He
            else:
                scale = np.sqrt(1.0 / fan_in)              # Xavier
            self.params[f"W{l}"] = rng.standard_normal((fan_out, fan_in)) * scale
            self.params[f"b{l}"] = np.zeros((fan_out, 1))

    def forward(self, X):
        """Propagation avant. Renvoie (sortie, cache) ; cache stocke A et Z."""
        act, _ = ACTIVATIONS[self.hidden_activation]
        cache = {"A0": X}
        A = X
        for l in range(1, self.L + 1):
            Z = self.params[f"W{l}"] @ A + self.params[f"b{l}"]
            A = softmax(Z) if l == self.L else act(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
        return A, cache

    def backward(self, Y, cache):
        """Rétropropagation (softmax + entropie croisée => dZ_L = A_L - Y)."""
        _, act_deriv = ACTIVATIONS[self.hidden_activation]
        grads = {}
        m = Y.shape[1]
        dZ = cache[f"A{self.L}"] - Y
        for l in range(self.L, 0, -1):
            A_prev = cache[f"A{l - 1}"]
            grads[f"dW{l}"] = (dZ @ A_prev.T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
            if l > 1:
                dA_prev = self.params[f"W{l}"].T @ dZ
                dZ = dA_prev * act_deriv(cache[f"Z{l - 1}"])
        return grads

    def predict_proba(self, X):
        A, _ = self.forward(X)
        return A

    # --- Sauvegarde / chargement : topologie ET poids (+ stats de normalisation) ---
    def save(self, path, mean=None, std=None):
        model = {
            "layer_sizes": np.array(self.layer_sizes),
            "hidden_activation": self.hidden_activation,
            "params": self.params,
            "mean": mean,
            "std": std,
        }
        np.save(path, model, allow_pickle=True)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True).item()
        net = cls(list(data["layer_sizes"]), str(data["hidden_activation"]))
        net.params = data["params"]
        return net, data.get("mean"), data.get("std")
