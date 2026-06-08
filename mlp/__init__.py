"""Package MLP : implémentation from scratch d'un perceptron multicouche (NumPy).

Aucune bibliothèque de réseau de neurones n'est utilisée : feedforward,
rétropropagation et descente de gradient sont codés à la main. NumPy ne sert
qu'à l'algèbre linéaire et matplotlib qu'à l'affichage des courbes.
"""

from .network import MLP
from .activations import ACTIVATIONS
from .optimizers import SGD, Adam, make_optimizer

__all__ = ["MLP", "ACTIVATIONS", "SGD", "Adam", "make_optimizer"]
