# Classification robuste — APS Failure at Scania Trucks

Projet de **classification binaire** sur un jeu de données industriel fortement **déséquilibré** : prédire les pannes du système de **pression d'air (APS)** des camions Scania à partir de 170 capteurs anonymisés.

Le pipeline couvre l'EDA, le prétraitement sans fuite de données, l'entraînement de trois familles de modèles, l'évaluation avec **métrique de coût métier** (challenge IDA 2016), la calibration des probabilités et l'interprétabilité **SHAP**.

---

## Dataset — APS Failure at Scania Trucks

| Propriété | Valeur |
|-----------|--------|
| **Source** | [UCI ML Repository — Dataset 421](https://archive.ics.uci.edu/dataset/421/aps+failure+and+operational+data+at+scania+trucks) |
| **Contexte** | Industrial Challenge, IDA 2016 |
| **Train** | 60 000 lignes (~59 000 `neg`, ~1 000 `pos`) |
| **Test** | 16 000 lignes (labels masqués pour la compétition) |
| **Features** | 170 capteurs + colonne `class` |
| **Coût métier** | Faux positif (FP) = **10** · Faux négatif (FN) = **500** |

Les fichiers CSV contiennent une **en-tête de licence GNU GPL** (~20 lignes) gérée automatiquement par `src/data/loader.py`.

### Téléchargement des données (obligatoire en local)

Les CSV **ne sont pas versionnés** sur GitHub (taille ~57 Mo). Après clonage du dépôt :

1. Télécharger l'archive sur [UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+and+operational+data+at+scania+trucks).
2. Extraire dans :

```text
data/raw/aps+failure+at+scania+trucks/
├── aps_failure_training_set.csv
├── aps_failure_test_set.csv
└── aps_failure_description.txt   # optionnel (déjà fourni dans le dépôt)
```

**Ne pas** conserver `aps+failure+at+scania+trucks.zip` dans `data/raw/` une fois les CSV extraits (doublon ~56 Mo).

---

## Structure du projet

```text
projet_ai_robuste/
├── data/
│   ├── raw/aps+failure+at+scania+trucks/   # CSV bruts (local uniquement)
│   └── processed/                            # Sorties notebook 01 (généré)
├── notebooks/                              # Pipeline principal (4 étapes)
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation_calibration.ipynb
│   └── 04_shap_analysis.ipynb
├── src/                                    # Code Python réutilisable
│   ├── data/loader.py                      # Chargement APS + coût métier
│   ├── preprocessing/pipeline.py           # Imputation, scaling, SMOTE (imblearn)
│   ├── evaluation/                         # Métriques, plots, calibration
│   ├── models/                             # Réservé (logique dans les notebooks)
│   └── utils/reproducibility.py            # Seeds, logging, CV
├── models/                                 # Modèles .pkl (généré, notebook 02–03)
├── reports/
│   ├── figures/                            # PNG 300 DPI (généré)
│   └── tables/                             # CSV d'export (ex. SHAP)
├── scripts/
│   ├── generate_all_notebooks.py           # Régénère les 4 notebooks
│   └── build_02_notebook.py                # Ancien générateur (référence)
├── requirements.txt
├── .gitignore
└── README.md
```

### Rôle des dossiers

| Dossier | Rôle |
|---------|------|
| `data/raw/` | Données source UCI (CSV + description). |
| `data/processed/` | `train_encoded.csv`, `test_encoded.csv`, `eda_results.joblib` (notebook 01). |
| `notebooks/` | Livrables et expérimentation reproductible. |
| `src/` | Modules importés par les notebooks (`sys.path` → racine projet). |
| `models/` | Artefacts `joblib` : régression logistique, RF, XGBoost, modèle calibré, base Optuna. |
| `reports/figures/` | Visualisations (EDA, courbes PR/ROC, calibration, SHAP). |
| `reports/tables/` | Tableaux exportés (importance SHAP, etc.). |
| `scripts/` | Génération / maintenance des notebooks. |

---

## Installation

**Prérequis** : Python 3.10+ recommandé.

```powershell
cd C:\Users\hp\projet_ai_robuste
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=aps-scania --display-name "Python (APS Scania)"
```

Sous Linux/macOS, remplacer l'activation par `source .venv/bin/activate`.

---

## Utilisation — ordre d'exécution

Exécuter les notebooks **dans l'ordre** depuis la racine du projet (kernel : environnement ci-dessus).

| Étape | Notebook | Contenu principal | Durée indicative |
|-------|----------|-------------------|------------------|
| **1** | `01_eda_preprocessing.ipynb` | EDA, corrélations, VIF, UMAP, comparaison `class_weight` vs SMOTE | 30–60 min |
| **2** | `02_model_training.ipynb` | Elastic Net, Random Forest + proximité, XGBoost + Optuna (100 essais) | 1–2 h |
| **3** | `03_evaluation_calibration.ipynb` | Test set, PR/ROC/MCC, coût total, calibration (ECE) | 20–40 min |
| **4** | `04_shap_analysis.ipynb` | SHAP TreeExplainer, dépendances, stabilité | 20–40 min |

**Dépendances entre notebooks**

- Le notebook **02** lit les données via `load_aps_data()` (pas besoin de `data/processed/` pour l'entraînement).
- Les notebooks **03** et **04** nécessitent les fichiers dans `models/` produits par le **02** :
  - `logreg_elasticnet.pkl`, `logreg_gridsearch.pkl`
  - `random_forest.pkl`
  - `xgboost_best.pkl`
- Le **03** peut produire `best_model_calibrated.pkl`.

Régénérer les notebooks 02–04 à partir du script :

```powershell
python scripts/generate_all_notebooks.py
```

---

## Méthodologie — les 4 étapes

### Étape 1 — EDA et préparation

- Analyse du déséquilibre (~1,67 % de positifs).
- Traitement des valeurs manquantes (`na`), outliers (IQR), multicolinéarité (Spearman, VIF).
- Visualisation (UMAP) et comparaison **pondération de classes** vs **SMOTE** (sans fuite : pipelines `imblearn`).

### Étape 2 — Modélisation

- **Baseline** : régression logistique Elastic Net (GridSearchCV).
- **Forêt aléatoire** : proximité OOB + projection MDS des outliers.
- **XGBoost** : optimisation Optuna (stratégies A/B), sauvegarde du meilleur pack.

**Métriques de validation** : F1-macro, AUPRC, MCC — **pas d'accuracy** (trompeuse sur classes rares).

### Étape 3 — Évaluation et calibration

- Courbes Precision–Recall, ROC, seuil optimal (MCC et **coût total** FP×10 + FN×500).
- Calibration (Platt / isotonic) et **ECE** (Expected Calibration Error).

### Étape 4 — Interprétabilité SHAP

- `TreeExplainer` sur le meilleur XGBoost.
- Summary, dependence plots, exemples waterfall.
- Export `reports/tables/shap_feature_importance.csv`.

---

## Résultats attendus

Après exécution complète du pipeline :

| Livrable | Emplacement |
|----------|-------------|
| Données encodées | `data/processed/train_encoded.csv`, `test_encoded.csv` |
| Modèles | `models/*.pkl`, `models/optuna_study.db` |
| Figures (300 DPI) | `reports/figures/*.png` (EDA, comparaison modèles, calibration, SHAP) |
| Table SHAP | `reports/tables/shap_feature_importance.csv` |

**Objectifs qualitatifs**

- Réduction du **coût total** sur le jeu de test par rapport aux baselines naïves (prédire toujours `neg`).
- Meilleur compromis rappel / précision sur la classe positive (pannes APS).
- Probabilités **calibrées** pour un seuil de décision métier fiable.
- Capteurs clés identifiés et interprétables via SHAP.

*Référence compétition* : meilleur score publié ~**9 920** (Costa & Nascimento, IDA 2016).

---

## Reproductibilité

- Seed global : **42** (`src/utils/reproducibility.py`).
- Validation croisée : `StratifiedKFold` (5 folds).
- Versions des bibliothèques loguées au démarrage de chaque notebook.

---

## Licence des données

Le dataset APS Scania est distribué sous **GNU General Public License v3** (voir `aps_failure_description.txt`). Respecter les conditions UCI/Scania pour toute redistribution.

---

## Auteur / contexte

Projet académique — **classification robuste** sur données industrielles déséquilibrées.
