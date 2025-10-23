Projet: MLP sur Wisconsin Diagnostic Breast Cancer (WDBC)

1) Objectif
- Prédire le diagnostic (M=malin, B=bénin) à partir de 30 features FNA avec un réseau de neurones (MLP).

2) Données
- Fichier: data.csv (format brut, sans en-têtes).
- 569 instances, 32 colonnes: id, diagnosis (B/M), + 30 features (mean, se, worst).
- Mapping cible: M -> 1, B -> 0.
- Pas de valeurs manquantes selon la description officielle.

3) Prétraitement
- Ajout des noms de colonnes attendus.
- Suppression de la colonne id.
- Encodage de diagnosis en 0/1.
- Standardisation des features avec StandardScaler (dans un pipeline).

4) Séparation des données
- Split entraînement/validation: 80% / 20%.
- Stratification sur la cible pour conserver le ratio de classes.
- graine (random_state) = 42 pour la reproductibilité.

5) Modèle (MLP)
- MLPClassifier (scikit-learn).
- hidden_layer_sizes = (64, 32)
- activation = relu, solver = adam
- alpha (L2) = 1e-4, learning_rate_init = 1e-3
- max_iter = 500, early_stopping = True, n_iter_no_change = 20

6) Entraînement & Évaluation
- Entraînement sur le set d’entraînement uniquement.
- Évaluation sur le set de validation.
- Métriques reportées: Accuracy, Precision, Recall, F1, ROC AUC.
- Option: courbe ROC.

7) Exécution
- Créer/activer l’environnement virtuel (.venv), puis:
  pip install -r requirements.txt
  python load-viz_data.py
- Adapter le séparateur à ton fichier:
  - sep=',' si CSV à virgules
  - sep=r'\s+' si colonnes séparées par espaces

8) Fichiers principaux
- load-viz_data.py: chargement, visualisations, split, entraînement MLP et évaluation.
- data.csv: données brutes.
- (optionnel) models/mlp_wdbc.joblib: modèle entraîné sauvegardé.

9) Références
- Wisconsin Diagnostic Breast Cancer (WDBC), UCI ML Repository.