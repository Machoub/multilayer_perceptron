# archive/ — fichiers obsolètes (hors livrable)

Ces fichiers proviennent de la **première version** du projet (exploration sur
Google Colab). Ils sont **conservés à titre de trace** mais ne font **pas**
partie du livrable évalué.

| Fichier | Remplacé par |
|---|---|
| `load-viz_data.ipynb` | `split.py` (découpage) + `split.py --visualize` (visualisation) |
| `mlp_train.ipynb` | `train.py` (entraînement) + `predict.py` (prédiction) + package `mlp/` |
| `train_data.csv`, `test_data.csv` | générés à la volée : `data_train.csv`, `data_valid.csv` |

> ⚠️ Les notebooks importent encore `scikit-learn` (`train_test_split`,
> `StandardScaler`). Le **livrable à la racine** (`split.py`, `train.py`,
> `predict.py`, `mlp/`) est lui **100 % from scratch** : le découpage stratifié
> et la standardisation sont réimplémentés en NumPy dans `mlp/data.py`.
