#!/usr/bin/env python3
"""Programme 2/3 : entraînement.

Réseau de neurones modulaire (taille des couches passée en argument), entraîné
par rétropropagation + descente de gradient (mini-lots). Affiche les métriques
train ET validation à chaque epoch, trace les deux courbes d'apprentissage, et
sauvegarde le modèle (topologie + poids + stats de normalisation).

Exemple (équivalent à l'exemple du sujet) :
    python train.py --layer 24 24 24 --epochs 84 --batch_size 8 \\
        --learning_rate 0.0314 --loss categoricalCrossentropy
"""

import argparse

import numpy as np

from mlp.data import (
    iterate_minibatches, load_dataset, standardize_apply, standardize_fit, to_one_hot,
)
from mlp.losses import LOSSES
from mlp.metrics import accuracy
from mlp.network import MLP
from mlp.optimizers import make_optimizer
from mlp.plotting import plot_learning_curves


def evaluate(net, X, Y, y, loss_fn):
    """Coût et accuracy sur un jeu complet."""
    A = net.predict_proba(X)
    return loss_fn(Y, A), accuracy(np.argmax(A, axis=0), y)


def main():
    p = argparse.ArgumentParser(description="Entraîne le MLP from scratch.")
    p.add_argument("--train", default="data_train.csv")
    p.add_argument("--valid", default="data_valid.csv")
    p.add_argument("--layer", type=int, nargs="+", default=[24, 24],
                   help="tailles des couches CACHÉES (>= 2 par défaut)")
    p.add_argument("--epochs", type=int, default=84)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.0314)
    p.add_argument("--activation", default="relu", choices=["relu", "sigmoid"])
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "momentum", "adam"])
    p.add_argument("--loss", default="categoricalCrossentropy", choices=list(LOSSES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stopping", action="store_true",
                   help="arrêt anticipé sur la val_loss (avec patience)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--out", default="saved_model.npy")
    p.add_argument("--plot", default="learning_curves.png")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if any(s < 1 for s in args.layer):
        p.error("--layer : les tailles de couches doivent être >= 1")
    if len(args.layer) < 2:
        print(f"⚠️  {len(args.layer)} couche cachée — le sujet recommande au moins 2.")

    # 1) Données (prétraitées par split.py ; on standardise ici, stats issues du train)
    X_train, y_train = load_dataset(args.train)
    X_valid, y_valid = load_dataset(args.valid)
    mean, std = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mean, std)
    X_valid = standardize_apply(X_valid, mean, std)
    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    # Vers la convention colonne = exemple, et cibles one-hot
    Xt, Xv = X_train.T, X_valid.T
    Yt, Yv = to_one_hot(y_train), to_one_hot(y_valid)

    # 2) Réseau modulaire : [n_features] + couches cachées + [2 classes]
    n_in, n_out = Xt.shape[0], 2
    layer_sizes = [n_in] + list(args.layer) + [n_out]
    net = MLP(layer_sizes, hidden_activation=args.activation, seed=args.seed)
    optimizer = make_optimizer(args.optimizer, args.learning_rate)
    # NB : pour une sortie softmax à 2 classes, 'binaryCrossentropy' et
    # 'categoricalCrossentropy' sont équivalents et résolvent tous deux vers la CCE.
    loss_fn = LOSSES[args.loss]
    print(f"topologie : {layer_sizes} | activation : {args.activation} | "
          f"optimiseur : {args.optimizer} | batch : {args.batch_size}")

    # 3) Boucle d'entraînement
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    rng = np.random.default_rng(args.seed)
    best_val, best_params, wait = float("inf"), None, 0
    width = len(str(args.epochs))

    for epoch in range(1, args.epochs + 1):
        for Xb, Yb in iterate_minibatches(Xt, Yt, args.batch_size, rng):
            _, cache = net.forward(Xb)
            grads = net.backward(Yb, cache)
            optimizer.update(net.params, grads)

        tr_loss, tr_acc = evaluate(net, Xt, Yt, y_train, loss_fn)
        va_loss, va_acc = evaluate(net, Xv, Yv, y_valid, loss_fn)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"epoch {epoch:0{width}d}/{args.epochs} - loss: {tr_loss:.4f} - "
              f"val_loss: {va_loss:.4f} - acc: {tr_acc:.4f} - val_acc: {va_acc:.4f}")

        if args.early_stopping:
            if va_loss < best_val - 1e-4:
                best_val, wait = va_loss, 0
                best_params = {k: v.copy() for k, v in net.params.items()}
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"> arrêt anticipé à l'epoch {epoch} "
                          f"(val_loss sans amélioration depuis {args.patience} epochs)")
                    break

    # On restaure les meilleurs poids — que l'arrêt soit anticipé ou que la boucle
    # aille jusqu'au bout (sinon on sauvegarderait les poids du dernier epoch).
    if args.early_stopping and best_params is not None:
        net.params = best_params

    # 4) Sauvegarde (topologie + poids + normalisation) et courbes
    net.save(args.out, mean=mean, std=std)
    print(f"> saving model '{args.out}' to disk...")
    if not args.no_plot:
        plot_learning_curves(history, args.plot)
        print(f"> courbes d'apprentissage sauvegardées dans '{args.plot}'")


if __name__ == "__main__":
    main()
