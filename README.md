# Multilayer Perceptron — WDBC (Wisconsin Diagnostic Breast Cancer)

Implémentation **from scratch** (sans bibliothèque de réseau de neurones) d'un perceptron multicouche
pour prédire si une tumeur est **maligne (M)** ou **bénigne (B)** à partir de 30
features. Feedforward, rétropropagation et descente de gradient sont codés à la
main ; aucune bibliothèque de réseau de neurones n'est utilisée. NumPy sert à
l'algèbre linéaire, pandas à la lecture des CSV et matplotlib aux courbes.

## Données
- `data.csv` : 569 exemples, 32 colonnes (id, diagnosis B/M, 30 features), brut,
  sans en-tête. Mapping cible : `M -> 1`, `B -> 0`.

## Installation
```bash
pip install -r requirements.txt      # numpy, pandas, matplotlib  (pas de sklearn)
# ou : make install
```

## Utilisation — 3 programmes
```bash
# 1) Découpage train/validation (stratifié, graine reproductible)
python split.py --data data.csv --val-size 0.2 --seed 42 [--visualize]

# 2) Entraînement (architecture modulaire via --layer)
python train.py --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314

# 3) Prédiction + évaluation (binary cross-entropy + métriques)
python predict.py --model saved_model.npy --data data_valid.csv
```
Ou via `make split && make train && make predict`.

### Architecture modulaire
La topologie n'est **pas** codée en dur : `--layer` définit les couches cachées
(au moins 2 par défaut). Les couches d'entrée (30) et de sortie (2) sont
déduites des données. Exemples :
```bash
python train.py --layer 24 24 24            # 3 couches cachées
python train.py --layer 16 8 --activation sigmoid --optimizer adam
```
Options : `--epochs`, `--batch_size`, `--learning_rate`, `--activation`
(relu/sigmoid), `--optimizer` (sgd/momentum/adam), `--loss`, `--seed`,
`--early-stopping --patience N`.

## Notions clés (soutenance)
- **Feedforward** : pour chaque couche `l`, `Z[l] = W[l]·A[l-1] + b[l]` puis
  `A[l] = activation(Z[l])`. Les couches cachées utilisent ReLU (ou sigmoid), la
  sortie utilise **softmax** pour produire une distribution de probabilité.
- **Backpropagation** : on calcule le gradient du coût par rapport à chaque
  poids via la règle de la chaîne, en partant de la sortie. Pour softmax +
  entropie croisée : `dZ[L] = A[L] - Y`, puis on remonte
  `dZ[l] = (W[l+1]ᵀ·dZ[l+1]) ⊙ activation'(Z[l])`.
- **Gradient descent** : `W -= learning_rate · dW` (SGD), appliqué par mini-lots.
  Variantes momentum et Adam disponibles.
- **Fonction d'erreur de prédiction (sujet)** :
  `E = -(1/N) Σ [ yₙ·log(pₙ) + (1-yₙ)·log(1-pₙ) ]` (binary cross-entropy).

La rétropropagation est validée par `python tests/gradient_check.py` (comparaison
aux différences finies, écart relatif < 1e-6).

## Fichiers
| Fichier | Rôle |
|---|---|
| `split.py` | découpage train/validation (+ visualisation) |
| `train.py` | entraînement, courbes, sauvegarde du modèle |
| `predict.py` | prédiction + évaluation (BCE, accuracy, precision/recall/F1) |
| `mlp/` | package partagé : `network`, `activations`, `losses`, `optimizers`, `data`, `metrics`, `plotting` |
| `tests/gradient_check.py` | vérification numérique de la backprop |
| `saved_model.npy` | modèle sauvegardé : **topologie + poids + normalisation** |
| `learning_curves.png` | courbes loss & accuracy (train + validation) |

## Bonus implémentés
- Optimiseurs momentum et **Adam**.
- **Early stopping** sur la validation loss (avec patience + restauration des
  meilleurs poids).
- Historique des métriques + courbes train/validation superposées.
- Métriques multiples : precision, recall, F1, matrice de confusion.

## Référence
Wisconsin Diagnostic Breast Cancer (WDBC), UCI Machine Learning Repository.
