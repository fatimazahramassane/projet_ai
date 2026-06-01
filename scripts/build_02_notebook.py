"""Generate notebooks/02_model_training.ipynb."""
import json
from pathlib import Path

cells = []


def md(text: str):
    lines = [line + "\n" for line in text.strip().split("\n")]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})


def code(text: str):
    lines = [line + "\n" for line in text.strip().split("\n")]
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": lines,
        }
    )


# --- Notebook content ---
md(
    """# Étape 2 : Entraînement des modèles (Livraison 2)

## Dataset : APS Failure at Scania Trucks

**Objectif** : Entraîner et comparer 3 familles de modèles sur une classification très déséquilibrée (~1,3 % de positifs).

### Modèles
1. **Régression logistique Elastic Net** (baseline) — `GridSearchCV`
2. **Random Forest** + matrice de proximité, MDS, `IsolationForest`
3. **XGBoost cost-sensitive** — Stratégie A (`scale_pos_weight`) vs B (Focal Loss) + Optuna TPE

### Métriques (jamais accuracy)
- F1-Macro, AUPRC, MCC (train et test)

---"""
)

md("## 0. Configuration et imports")

code(
    """
import os
import sys
import time
import warnings
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.manifold import MDS
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_aps_data, get_X_y
from src.preprocessing.pipeline import create_model_pipeline, verify_no_leakage
from src.evaluation.metrics import compute_all_metrics, print_metrics_report, find_optimal_threshold
from src.evaluation.plots import setup_plot_style, save_figure, COLORS
from src.utils.reproducibility import set_all_seeds, log_environment_info, setup_logging, get_cv_splitter

warnings.filterwarnings('ignore')
setup_logging(level=logging.INFO)
setup_plot_style()
%matplotlib inline

SEED = 42
set_all_seeds(SEED)
N_FOLDS = 5
cv = get_cv_splitter(n_splits=N_FOLDS, seed=SEED)

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f'Racine projet : {PROJECT_ROOT}')
log_environment_info()
"""
)

md(
    """## 1. Chargement des données

Séparation train/test officielle du challenge IDA 2016. Le prétraitement (imputation, scaling) est encapsulé dans les pipelines pour éviter toute fuite."""
)

code(
    """
train_df, test_df = load_aps_data(project_root=PROJECT_ROOT)
X_train, y_train = get_X_y(train_df)
X_test, y_test = get_X_y(test_df)

verify_no_leakage(X_train.values, X_test.values, y_train.values, y_test.values)

n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())
SCALE_POS_WEIGHT = n_neg / n_pos

print(f'Train : {X_train.shape} | Positifs : {n_pos} ({y_train.mean()*100:.2f}%)')
print(f'Test  : {X_test.shape} | Positifs : {int(y_test.sum())}')
print(f'scale_pos_weight = {SCALE_POS_WEIGHT:.1f}')
"""
)

md("## 2. Fonctions utilitaires")

code(
    """
scoring_metrics = {
    'f1_macro': 'f1_macro',
    'mcc': make_scorer(matthews_corrcoef),
    'auprc': make_scorer(average_precision_score, needs_proba=True),
}


def get_proba(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate_model(model, X, y, threshold: float = 0.5) -> Dict[str, float]:
    y_proba = get_proba(model, X)
    y_pred = (y_proba >= threshold).astype(int)
    y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
    return compute_all_metrics(y_arr, y_pred, y_proba)


def metrics_row(model_name: str, split: str, metrics: Dict) -> dict:
    return {
        'Modèle': model_name,
        'Jeu': split,
        'F1-Macro': round(metrics['F1-Macro'], 4),
        'AUPRC': round(metrics.get('AUPRC', np.nan), 4),
        'MCC': round(metrics['MCC'], 4),
        'Coût IDA': int(metrics['Cost']),
    }


def compute_proximity_matrix(leaf_indices: np.ndarray) -> np.ndarray:
    n_samples, n_trees = leaf_indices.shape
    prox = np.zeros((n_samples, n_samples), dtype=np.float64)
    for t in range(n_trees):
        leaves = leaf_indices[:, t]
        for leaf_id in np.unique(leaves):
            idx = np.where(leaves == leaf_id)[0]
            if len(idx) > 1:
                prox[np.ix_(idx, idx)] += 1.0
    prox /= n_trees
    np.fill_diagonal(prox, 1.0)
    return prox


def focal_loss_objective(gamma: float = 2.0, alpha: float = 0.25):
    def objective(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        pt = np.where(y == 1, p, 1 - p)
        w = np.where(y == 1, alpha, 1 - alpha)
        modulating = (1.0 - pt) ** gamma
        grad = w * modulating * (p - y)
        hess = np.maximum(w * modulating * p * (1.0 - p), 1e-6)
        return grad, hess
    return objective


all_results = []
optimal_thresholds = {}
"""
)

md(
    """---

## 3. Modèle 1 : Régression logistique Elastic Net (baseline)

**Pipeline** : `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced')`

**Hyperparamètres** : `C` et `l1_ratio` via `GridSearchCV` (5 folds stratifiés, `refit='mcc'`).

**Justification** : Le scaling est indispensable pour la régression logistique ; `class_weight='balanced'` compense le déséquilibre sans sur-échantillonner artificiellement."""
)

code(
    """
log_reg = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    class_weight='balanced',
    max_iter=5000,
    random_state=SEED,
)

pipe_lr = create_model_pipeline(log_reg, with_smote=False, scaling=True)

param_grid_lr = {
    'model__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'model__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
}

print('GridSearchCV — Elastic Net (refit=MCC)...')
t0 = time.time()
grid_lr = GridSearchCV(
    pipe_lr,
    param_grid_lr,
    cv=cv,
    scoring=scoring_metrics,
    refit='mcc',
    n_jobs=-1,
    verbose=1,
)
grid_lr.fit(X_train, y_train)
print(f'Durée : {time.time() - t0:.1f}s')
print(f'Meilleurs params : {grid_lr.best_params_}')
print(f'CV MCC : {grid_lr.best_score_:.4f}')

best_lr = grid_lr.best_estimator_
y_proba_lr_tr = get_proba(best_lr, X_train)
thr_lr, _, _, _ = find_optimal_threshold(y_train.values, y_proba_lr_tr, metric='mcc')
optimal_thresholds['Elastic Net'] = thr_lr

m_lr_tr = evaluate_model(best_lr, X_train, y_train, thr_lr)
m_lr_te = evaluate_model(best_lr, X_test, y_test, thr_lr)
all_results.append(metrics_row('Elastic Net', 'Train', m_lr_tr))
all_results.append(metrics_row('Elastic Net', 'Test', m_lr_te))
print_metrics_report(y_test.values, (get_proba(best_lr, X_test) >= thr_lr).astype(int),
                     get_proba(best_lr, X_test), model_name='Elastic Net (test)')
"""
)

md(
    """---

## 4. Modèle 2 : Random Forest + proximité + MDS + outliers

- `RandomForestClassifier(n_estimators=500, max_depth=None, class_weight='balanced')`
- **Matrice de proximité** : fréquence de co-occurrence dans la même feuille terminale
- **MDS** : projection 2D de la matrice de dissimilarité `1 - proximité`
- **IsolationForest** (`contamination=0.05`) sur l'embedding MDS

> **Note mémoire** : une matrice 60k×60k (~28 Go) est irréalisable. On entraîne le RF sur tout le train, puis on calcule proximité/MDS sur un **sous-échantillon stratifié** (5 000 observations) pour la visualisation et l'analyse des outliers."""
)

code(
    """
rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight='balanced',
        n_jobs=-1,
        random_state=SEED,
    )),
])

print('Entraînement Random Forest (500 arbres)...')
t0 = time.time()
rf_model.fit(X_train, y_train)
print(f'Durée : {time.time() - t0:.1f}s')

# Seuil optimal sur train
y_proba_rf_tr = get_proba(rf_model, X_train)
thr_rf, _, _, _ = find_optimal_threshold(y_train.values, y_proba_rf_tr, metric='mcc')
optimal_thresholds['Random Forest'] = thr_rf

m_rf_tr = evaluate_model(rf_model, X_train, y_train, thr_rf)
m_rf_te = evaluate_model(rf_model, X_test, y_test, thr_rf)
all_results.append(metrics_row('Random Forest', 'Train', m_rf_tr))
all_results.append(metrics_row('Random Forest', 'Test', m_rf_te))
print_metrics_report(y_test.values, (get_proba(rf_model, X_test) >= thr_rf).astype(int),
                     get_proba(rf_model, X_test), model_name='Random Forest (test)')
"""
)

code(
    """
PROXIMITY_SAMPLE_SIZE = 5000
rng = np.random.RandomState(SEED)

idx_pos = np.where(y_train.values == 1)[0]
idx_neg = np.where(y_train.values == 0)[0]
n_pos_s = min(len(idx_pos), int(PROXIMITY_SAMPLE_SIZE * y_train.mean()))
n_neg_s = PROXIMITY_SAMPLE_SIZE - n_pos_s
idx_sample = np.concatenate([
    rng.choice(idx_pos, n_pos_s, replace=False),
    rng.choice(idx_neg, n_neg_s, replace=False),
])
rng.shuffle(idx_sample)

X_sub = X_train.iloc[idx_sample]
y_sub = y_train.iloc[idx_sample]
X_sub_imp = rf_model.named_steps['imputer'].transform(X_sub)
rf_fitted = rf_model.named_steps['rf']

print(f'Sous-échantillon proximité : {len(idx_sample)} (pos={n_pos_s}, neg={n_neg_s})')
leaf_idx = rf_fitted.apply(X_sub_imp)
proximity = compute_proximity_matrix(leaf_idx)
dissimilarity = 1.0 - proximity

print('MDS 2D...')
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED, n_init=4, max_iter=300)
embedding = mds.fit_transform(dissimilarity)
print(f'Stress MDS : {mds.stress_:.4f}')
"""
)

code(
    """
y_proba_sub = rf_fitted.predict_proba(X_sub_imp)[:, 1]
y_pred_sub = (y_proba_sub >= thr_rf).astype(int)
uncertainty = np.abs(y_proba_sub - 0.5)

iso = IsolationForest(contamination=0.05, random_state=SEED)
outlier_labels = iso.fit_predict(embedding)
is_outlier = outlier_labels == -1

print(f'Outliers détectés (IsolationForest) : {is_outlier.sum()} ({is_outlier.mean()*100:.1f}%)')

fig, ax = plt.subplots(figsize=(12, 8))
mask_in = ~is_outlier
ax.scatter(embedding[mask_in, 0], embedding[mask_in, 1], c=COLORS['neg_class'],
           alpha=0.25, s=12, label='Inliers')
ax.scatter(embedding[is_outlier, 0], embedding[is_outlier, 1], c=COLORS['danger'],
           alpha=0.9, s=40, edgecolors='black', linewidths=0.5, label='Outliers (IF)')
ax.set_title('MDS 2D de la matrice de proximité RF — outliers en rouge')
ax.set_xlabel('MDS 1')
ax.set_ylabel('MDS 2')
ax.legend()
plt.tight_layout()
save_figure(fig, 'rf_mds_proximity_outliers', save_dir=FIGURES_DIR)
plt.show()
"""
)

code(
    """
outlier_df = X_sub.copy()
outlier_df['y_true'] = y_sub.values
outlier_df['y_pred'] = y_pred_sub
outlier_df['proba_pos'] = y_proba_sub
outlier_df['uncertainty'] = uncertainty
outlier_df['is_outlier'] = is_outlier
outlier_df['misclassified'] = outlier_df['y_true'] != outlier_df['y_pred']

print('=== Statistiques outliers vs inliers ===')
for col in ['uncertainty', 'proba_pos']:
    print(f"{col} — inliers: {outlier_df.loc[~is_outlier, col].mean():.3f} | "
          f"outliers: {outlier_df.loc[is_outlier, col].mean():.3f}")

print(f"Taux de mauvaise classification — outliers: {outlier_df.loc[is_outlier, 'misclassified'].mean()*100:.1f}%")
print(f"Taux de mauvaise classification — inliers: {outlier_df.loc[~is_outlier, 'misclassified'].mean()*100:.1f}%")

# Features les plus divergentes (z-score moyen outliers vs global)
global_median = X_train.median()
outlier_median = outlier_df.loc[is_outlier, X_train.columns].median()
z_shift = ((outlier_median - global_median) / (X_train.std() + 1e-8)).abs().sort_values(ascending=False)
top_features = z_shift.head(10)
print('\\nTop 10 capteurs décalés chez les outliers (|z| vs médiane train) :')
display(top_features.to_frame('z_shift'))

print('''
**Interprétation** :
- **Incertitude élevée** (`|p - 0.5|` faible) → frontière de décision floue, pas assez de votes cohérents dans le RF.
- **Erreurs de classification** concentrées → cas limites ou bruit de capteur.
- **Capteurs atypiques** (fort z_shift) → patterns rares non vus souvent dans les feuilles « normales ».
- Les outliers MDS sont souvent **entre les grappes** de proximité : le modèle n'a pas de voisins RF stables.
''')
"""
)

md(
    """---

## 5. Modèle 3 : XGBoost + Optuna (TPE)

Deux stratégies comparées :
- **A** : `scale_pos_weight` ∈ {10, 30, 59, 77} — pondération explicite des positifs
- **B** : **Focal Loss** (γ=2, α=0.25) — focus sur exemples difficiles

**Objectif Optuna** : maximiser le **MCC** sur le fold de validation (5 folds stratifiés).

### Bornes de l'espace de recherche
| Paramètre | Borne | Justification |
|-----------|-------|---------------|
| `max_depth` | 3–8 | Limite la complexité (170 features, risque de sur-apprentissage) |
| `learning_rate` | 0.01–0.3 (log) | Contrôle la vitesse d'apprentissage |
| `lambda` | 1e-3–10 (log) | Régularisation L2 |
| `alpha` | 1e-3–10 (log) | Régularisation L1 |
| `subsample` | 0.6–1.0 | Bagging — robustesse |
| `colsample_bytree` | 0.6–1.0 | Réduit la corrélation entre arbres |"""
)

code(
    """
_imputer_xgb = SimpleImputer(strategy='median')
X_tr_imp = _imputer_xgb.fit_transform(X_train)
X_te_imp = _imputer_xgb.transform(X_test)

SPW_CANDIDATES = [10, 30, int(round(SCALE_POS_WEIGHT)), 77]
N_TRIALS = 50


def run_xgb_cv_mcc(X, y, params, use_focal: bool = False) -> float:
    dtrain_full = xgb.DMatrix(X, label=y)
    mcc_folds = []
    for tr_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        booster_params = {
            'max_depth': params['max_depth'],
            'eta': params['learning_rate'],
            'lambda': params['lambda'],
            'alpha': params['alpha'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': SEED,
            'verbosity': 0,
        }
        if not use_focal:
            booster_params['scale_pos_weight'] = params['scale_pos_weight']
        train_kw = {
            'num_boost_round': params.get('n_estimators', 300),
            'evals': [(dval, 'val')],
            'verbose_eval': False,
        }
        if use_focal:
            bst = xgb.train(
                booster_params, dtrain,
                obj=focal_loss_objective(gamma=2.0, alpha=0.25),
                **train_kw,
            )
        else:
            bst = xgb.train(booster_params, dtrain, **train_kw)
        proba = bst.predict(dval)
        pred = (proba >= 0.5).astype(int)
        mcc_folds.append(matthews_corrcoef(y_val, pred))
    return float(np.mean(mcc_folds))
"""
)

code(
    """
def create_objective(strategy: str):
    use_focal = strategy == 'B'

    def objective(trial: optuna.Trial) -> float:
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        }
        if not use_focal:
            params['scale_pos_weight'] = trial.suggest_categorical(
                'scale_pos_weight', SPW_CANDIDATES
            )
        return run_xgb_cv_mcc(X_tr_imp, y_train.values, params, use_focal=use_focal)

    return objective


print('=== Optuna Stratégie A (scale_pos_weight) ===')
study_a = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED),
    study_name='xgb_scale_pos_weight',
)
study_a.optimize(create_objective('A'), n_trials=N_TRIALS, show_progress_bar=True)
print(f'Meilleur MCC CV : {study_a.best_value:.4f}')
print(f'Meilleurs params : {study_a.best_params}')
"""
)

code(
    """
print('=== Optuna Stratégie B (Focal Loss) ===')
study_b = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED),
    study_name='xgb_focal_loss',
)
study_b.optimize(create_objective('B'), n_trials=N_TRIALS, show_progress_bar=True)
print(f'Meilleur MCC CV : {study_b.best_value:.4f}')
print(f'Meilleurs params : {study_b.best_params}')
"""
)

code(
    """
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)

def _save_optuna_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    try:
        fig.write_image(path)
        print(f'  Sauvegardé : {path}')
    except Exception as e:
        html_path = path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f'  PNG indisponible ({e}) — HTML : {html_path}')
    return fig

fig_hist_a = plot_optimization_history(study_a)
fig_hist_a.update_layout(title='Optuna — Historique (Stratégie A)')
_save_optuna_fig(fig_hist_a, 'optuna_history_strategy_a.png')

fig_imp_a = plot_param_importances(study_a)
fig_imp_a.update_layout(title='Optuna — Importance des hyperparamètres (A)')
_save_optuna_fig(fig_imp_a, 'optuna_importance_strategy_a.png')

fig_slice_a = plot_slice(study_a, params=['max_depth', 'learning_rate'])
fig_slice_a.update_layout(title='Optuna — Slice max_depth & learning_rate (A)')
_save_optuna_fig(fig_slice_a, 'optuna_slice_strategy_a.png')

fig_hist_b = plot_optimization_history(study_b)
fig_hist_b.update_layout(title='Optuna — Historique (Stratégie B)')
_save_optuna_fig(fig_hist_b, 'optuna_history_strategy_b.png')

fig_imp_b = plot_param_importances(study_b)
_save_optuna_fig(fig_imp_b, 'optuna_importance_strategy_b.png')

fig_slice_b = plot_slice(study_b, params=['max_depth', 'learning_rate'])
_save_optuna_fig(fig_slice_b, 'optuna_slice_strategy_b.png')

fig_hist_a.show()
fig_imp_a.show()
fig_slice_a.show()
print('Figures Optuna traitées.')
"""
)

code(
    """
def train_final_xgb(best_params: dict, use_focal: bool, strategy_name: str):
    params = best_params.copy()
    n_rounds = params.pop('n_estimators', 300)
    dtrain = xgb.DMatrix(X_tr_imp, label=y_train.values)
    dtest = xgb.DMatrix(X_te_imp, label=y_test.values)
    booster_params = {
        'max_depth': params['max_depth'],
        'eta': params['learning_rate'],
        'lambda': params['lambda'],
        'alpha': params['alpha'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'objective': 'binary:logistic',
        'seed': SEED,
        'verbosity': 0,
    }
    if not use_focal:
        booster_params['scale_pos_weight'] = params['scale_pos_weight']
    if use_focal:
        bst = xgb.train(
            booster_params, dtrain, num_boost_round=n_rounds,
            obj=focal_loss_objective(gamma=2.0, alpha=0.25),
        )
    else:
        bst = xgb.train(booster_params, dtrain, num_boost_round=n_rounds)
    proba_tr = bst.predict(dtrain)
    proba_te = bst.predict(dtest)
    thr, _, _, _ = find_optimal_threshold(y_train.values, proba_tr, metric='mcc')
    pred_tr = (proba_tr >= thr).astype(int)
    pred_te = (proba_te >= thr).astype(int)
    m_tr = compute_all_metrics(y_train.values, pred_tr, proba_tr)
    m_te = compute_all_metrics(y_test.values, pred_te, proba_te)
    optimal_thresholds[strategy_name] = thr
    return bst, m_tr, m_te, thr


bst_a, m_xgb_a_tr, m_xgb_a_te, thr_a = train_final_xgb(
    study_a.best_params, use_focal=False, strategy_name='XGBoost A (SPW)'
)
bst_b, m_xgb_b_tr, m_xgb_b_te, thr_b = train_final_xgb(
    study_b.best_params, use_focal=True, strategy_name='XGBoost B (Focal)'
)

all_results.append(metrics_row('XGBoost A (scale_pos_weight)', 'Train', m_xgb_a_tr))
all_results.append(metrics_row('XGBoost A (scale_pos_weight)', 'Test', m_xgb_a_te))
all_results.append(metrics_row('XGBoost B (Focal Loss)', 'Train', m_xgb_b_tr))
all_results.append(metrics_row('XGBoost B (Focal Loss)', 'Test', m_xgb_b_te))

print(f'\\nStratégie A — test MCC={m_xgb_a_te["MCC"]:.4f}, AUPRC={m_xgb_a_te["AUPRC"]:.4f}')
print(f'Stratégie B — test MCC={m_xgb_b_te["MCC"]:.4f}, AUPRC={m_xgb_b_te["AUPRC"]:.4f}')
winner = 'A' if m_xgb_a_te['MCC'] >= m_xgb_b_te['MCC'] else 'B'
print(f'\\n🏆 Meilleure stratégie XGBoost sur test (MCC) : Stratégie {winner}')
"""
)

md("## 6. Tableau comparatif des modèles")

code(
    """
results_df = pd.DataFrame(all_results)
results_pivot = results_df.pivot(index='Modèle', columns='Jeu', values=['F1-Macro', 'AUPRC', 'MCC'])
results_pivot = results_pivot.sort_index()

print('=' * 70)
print('TABLEAU COMPARATIF — F1-Macro | AUPRC | MCC (Train & Test)')
print('=' * 70)
display(results_df.style.background_gradient(subset=['MCC'], cmap='YlGn').format({
    'F1-Macro': '{:.4f}', 'AUPRC': '{:.4f}', 'MCC': '{:.4f}'
}))
display(results_pivot)

# Meilleur modèle global sur test (MCC)
test_rows = results_df[results_df['Jeu'] == 'Test'].copy()
best_model = test_rows.loc[test_rows['MCC'].idxmax()]
print(f"\\n✅ Meilleur modèle (MCC test) : {best_model['Modèle']} — MCC={best_model['MCC']:.4f}")
"""
)

code(
    """
fig, ax = plt.subplots(figsize=(10, 6))
models_plot = test_rows['Modèle'].tolist()
x = np.arange(len(models_plot))
width = 0.25
ax.bar(x - width, test_rows['F1-Macro'], width, label='F1-Macro', color=COLORS['primary'])
ax.bar(x, test_rows['AUPRC'], width, label='AUPRC', color=COLORS['secondary'])
ax.bar(x + width, test_rows['MCC'], width, label='MCC', color=COLORS['success'])
ax.set_xticks(x)
ax.set_xticklabels(models_plot, rotation=25, ha='right')
ax.set_ylabel('Score')
ax.set_title('Comparaison des modèles — métriques sur le jeu TEST')
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
save_figure(fig, 'model_comparison_test_metrics', save_dir=FIGURES_DIR)
plt.show()
"""
)

md(
    """## 7. Sauvegarde des modèles et synthèse

Modèles sérialisés dans `models/` pour l'étape 3 (calibration & évaluation)."""
)

code(
    """
import joblib

joblib.dump(best_lr, os.path.join(MODELS_DIR, 'elastic_net_best.joblib'))
joblib.dump(grid_lr, os.path.join(MODELS_DIR, 'elastic_net_gridsearch.joblib'))
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_500.joblib'))
bst_a.save_model(os.path.join(MODELS_DIR, 'xgboost_strategy_a.json'))
bst_b.save_model(os.path.join(MODELS_DIR, 'xgboost_strategy_b_focal.json'))
joblib.dump({'study_a': study_a.best_params, 'study_b': study_b.best_params,
             'thresholds': optimal_thresholds},
            os.path.join(MODELS_DIR, 'training_metadata.joblib'))

reports_dir = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(reports_dir, exist_ok=True)
results_df.to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)
print('Modèles et métadonnées sauvegardés.')
print('\\n--- SYNTHÈSE LIVRAISON 2 ---')
print('1. Elastic Net : baseline interprétable, sensible au scaling.')
print('2. Random Forest : proximité/MDS révèle les cas ambigus (outliers IF).')
print('3. XGBoost : Optuna 50 essais — comparer stratégies A (SPW) vs B (Focal).')
print('Prochaine étape : notebooks/03_evaluation_calibration.ipynb')
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "02_model_training.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Written {out} ({len(cells)} cells)")
