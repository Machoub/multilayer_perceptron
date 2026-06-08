"""Chargement, prétraitement et découpage des données (NumPy/pandas, sans sklearn)."""

import numpy as np
import pandas as pd

# Noms de colonnes du jeu WDBC brut (32 colonnes : id, diagnosis, 30 features).
COLUMNS = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
FEATURE_NAMES = COLUMNS[2:]


def load_raw(path):
    """Lit le data.csv brut (sans en-tête) : nomme les colonnes, retire id,
    encode diagnosis M->1 / B->0. Renvoie un DataFrame (diagnosis + 30 features)."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Attendu {len(COLUMNS)} colonnes, reçu {df.shape[1]}")
    df = df.drop(columns=["id"])
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    if df["diagnosis"].isna().any():
        raise ValueError("colonne diagnosis : valeurs autres que M/B détectées")
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df


def load_dataset(path):
    """Lit un CSV déjà découpé (en-tête : diagnosis + features).
    Renvoie X (m, n_features) et y (m,) dans {0,1}."""
    df = pd.read_csv(path)
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    if df["diagnosis"].isna().any():
        raise ValueError("colonne diagnosis : valeurs autres que M/B détectées")
    y = df["diagnosis"].to_numpy().astype(int)
    X = df.drop(columns=["diagnosis"]).to_numpy(dtype=float)
    return X, y


def stratified_split(y, val_size, seed):
    """Découpe stratifié (conserve le ratio des classes). Renvoie (idx_train, idx_val)."""
    rng = np.random.default_rng(seed)
    idx_train, idx_val = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_size))
        idx_val.extend(idx[:n_val])
        idx_train.extend(idx[n_val:])
    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    return idx_train, idx_val


def standardize_fit(X):
    """Calcule moyenne/écart-type par feature (sur le train uniquement)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


def to_one_hot(y, n_classes=2):
    """y (m,) -> (n_classes, m) one-hot, via np.eye."""
    return np.eye(n_classes)[y].T


def iterate_minibatches(X, Y, batch_size, rng):
    """Itère sur des mini-lots de colonnes (exemples). batch_size None/<=0 ou
    >= m => un seul lot mélangé (full-batch)."""
    m = X.shape[1]
    idx = rng.permutation(m)
    if not batch_size or batch_size >= m:
        yield X[:, idx], Y[:, idx]
        return
    for start in range(0, m, batch_size):
        b = idx[start:start + batch_size]
        yield X[:, b], Y[:, b]
