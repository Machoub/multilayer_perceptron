#!/usr/bin/env python3
"""Programme 1/3 : découpage du jeu de données.

Lit le fichier brut (data.csv), le prétraite (nommage des colonnes, suppression
de l'id, encodage diagnosis M->1 / B->0) puis le découpe en deux parties
stratifiées — entraînement et validation — écrites en CSV. Une graine assure
la reproductibilité. Option --visualize pour un aperçu des features par classe.

Exemple :
    python split.py --data data.csv --val-size 0.2 --seed 42
"""

import argparse

from mlp.data import FEATURE_NAMES, load_raw, stratified_split


def visualize(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = FEATURE_NAMES
    n_cols = 6
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.6))
    axes = axes.flatten()
    malin = df[df["diagnosis"] == 1]
    benin = df[df["diagnosis"] == 0]
    for i, col in enumerate(features):
        ax = axes[i]
        ax.hist(benin[col], bins=25, alpha=0.6, label="Benign", color="tab:blue")
        ax.hist(malin[col], bins=25, alpha=0.6, label="Malignant", color="tab:red")
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(features), len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"> histogrammes par classe sauvegardés dans '{out_path}'")


def main():
    p = argparse.ArgumentParser(description="Découpe le jeu WDBC en train/validation.")
    p.add_argument("--data", default="data.csv", help="fichier brut (sans en-tête)")
    p.add_argument("--val-size", type=float, default=0.2, help="proportion de validation")
    p.add_argument("--seed", type=int, default=42, help="graine de reproductibilité")
    p.add_argument("--out-train", default="data_train.csv")
    p.add_argument("--out-valid", default="data_valid.csv")
    p.add_argument("--visualize", action="store_true", help="sauvegarde des histogrammes par classe")
    args = p.parse_args()

    df = load_raw(args.data)
    print(f"jeu chargé : {df.shape[0]} exemples, {df.shape[1] - 1} features")
    print(f"  malins (M=1) : {(df['diagnosis'] == 1).sum()} | "
          f"bénins (B=0) : {(df['diagnosis'] == 0).sum()}")

    if args.visualize:
        visualize(df, "data_features.png")

    idx_train, idx_val = stratified_split(df["diagnosis"].to_numpy(), args.val_size, args.seed)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_valid = df.iloc[idx_val].reset_index(drop=True)

    df_train.to_csv(args.out_train, index=False)
    df_valid.to_csv(args.out_valid, index=False)
    print(f"> '{args.out_train}' : {df_train.shape[0]} exemples")
    print(f"> '{args.out_valid}' : {df_valid.shape[0]} exemples")


if __name__ == "__main__":
    main()
