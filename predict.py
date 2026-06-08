#!/usr/bin/env python3
"""Programme 3/3 : prédiction et évaluation.

Charge le modèle sauvegardé (topologie + poids + normalisation), prédit sur un
jeu fourni puis l'évalue avec l'entropie croisée binaire (formule du sujet),
plus accuracy, precision, recall, F1 et matrice de confusion (métriques bonus).

Exemple :
    python predict.py --model saved_model.npy --data data_valid.csv
"""

import argparse

from mlp.data import load_dataset, standardize_apply
from mlp.losses import binary_cross_entropy
from mlp.metrics import accuracy, confusion, precision_recall_f1
from mlp.network import MLP

LABELS = {0: "Benign (B)", 1: "Malignant (M)"}


def main():
    p = argparse.ArgumentParser(description="Prédit et évalue avec le MLP entraîné.")
    p.add_argument("--model", default="saved_model.npy")
    p.add_argument("--data", default="data_valid.csv")
    p.add_argument("--preview", type=int, default=10, help="nb de prédictions affichées")
    args = p.parse_args()

    net, mean, std = MLP.load(args.model)
    X, y_true = load_dataset(args.data)
    if mean is not None:
        X = standardize_apply(X, mean, std)  # mêmes stats que l'entraînement

    proba = net.predict_proba(X.T)        # (2, m) : [P(bénin), P(malin)]
    y_pred = proba.argmax(axis=0)
    p_malin = proba[1, :]

    bce = binary_cross_entropy(y_true, p_malin)
    acc = accuracy(y_pred, y_true)
    precision, recall, f1 = precision_recall_f1(y_pred, y_true)
    tp, tn, fp, fn = confusion(y_pred, y_true)

    print("=" * 44)
    print(f"  Binary cross-entropy : {bce:.4f}")
    print(f"  Accuracy             : {acc:.4f}")
    print(f"  Precision (malin)    : {precision:.4f}")
    print(f"  Recall    (malin)    : {recall:.4f}")
    print(f"  F1-score  (malin)    : {f1:.4f}")
    print("-" * 44)
    print(f"  Matrice de confusion : TP={tp} TN={tn} FP={fp} FN={fn}")
    print("=" * 44)

    n = min(args.preview, len(y_true))
    if n > 0:
        print(f"\nAperçu des {n} premières prédictions :")
        print(f"{'Index':<7}{'Prédit':<16}{'Réel':<16}{'P(malin)':<10}{'OK'}")
        print("-" * 52)
        for i in range(n):
            ok = "OK" if y_pred[i] == y_true[i] else "X"
            print(f"{i:<7}{LABELS[y_pred[i]]:<16}{LABELS[y_true[i]]:<16}"
                  f"{p_malin[i]:<10.3f}{ok}")


if __name__ == "__main__":
    main()
