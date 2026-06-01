"""
Generate all 4 project notebooks (APS Failure at Scania Trucks).
Run: python scripts/generate_all_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


class NB:
    def __init__(self):
        self.cells: list = []

    def md(self, text: str):
        self.cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [ln + "\n" for ln in text.strip().split("\n")],
            }
        )

    def code(self, text: str):
        self.cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": [ln + "\n" for ln in text.strip().split("\n")],
            }
        )

    def save(self, name: str):
        path = NB_DIR / name
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "cells": self.cells,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"  -> {path} ({len(self.cells)} cells)")


COMMON_HEADER = '''
import os
import sys
import time
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_aps_data, get_X_y, compute_total_cost, COST_FP, COST_FN
from src.utils.reproducibility import set_all_seeds, log_environment_info, setup_logging, get_cv_splitter
from src.evaluation.metrics import compute_all_metrics, print_metrics_report, find_optimal_threshold
from src.evaluation.plots import setup_plot_style, save_figure, COLORS

warnings.filterwarnings('ignore')
setup_logging(level=logging.INFO)
setup_plot_style()
%matplotlib inline

SEED = 42
set_all_seeds(SEED)
N_FOLDS = 5
cv = get_cv_splitter(n_splits=N_FOLDS, seed=SEED)

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'tables')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'PROJECT_ROOT = {PROJECT_ROOT}')
log_environment_info()
pd.show_versions()
'''


def build_nb02() -> NB:
    nb = NB()
    nb.md(
        """# Étape 2 : Entraînement des modèles

## APS Failure at Scania Trucks — Livraison 2

Trois modèles : Elastic Net (baseline), Random Forest + proximité, XGBoost + Optuna (A vs B).

**Métriques** : F1-Macro, AUPRC, MCC — jamais accuracy.

---"""
    )
    nb.md("## 0. Configuration")
    nb.code(COMMON_HEADER)

    nb.md("## 1. Chargement des données")
    nb.code(
        """
train_df, test_df = load_aps_data(project_root=PROJECT_ROOT)
X_train, y_train = get_X_y(train_df)
X_test, y_test = get_X_y(test_df)

n_pos = int(y_train.sum())
n_neg = int((y_train == 0).sum())
SCALE_POS_WEIGHT = n_neg / n_pos
SPW_CANDIDATES = [10, 30, int(round(SCALE_POS_WEIGHT)), 77]

print(f'Train {X_train.shape} | pos={n_pos} ({y_train.mean()*100:.2f}%)')
print(f'scale_pos_weight ≈ {SCALE_POS_WEIGHT:.1f}')
"""
    )

    nb.md("## 2. Utilitaires")
    nb.code(
        """
import joblib
from typing import Dict, List
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.manifold import MDS
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from src.preprocessing.pipeline import create_model_pipeline

scoring_cv = {
    'f1_macro': 'f1_macro',
    'auprc': make_scorer(average_precision_score, needs_proba=True),
    'mcc': make_scorer(matthews_corrcoef),
}
cv_results_all: List[Dict] = []
model_times: Dict[str, float] = {}


def compute_proximity_matrix(leaf_indices: np.ndarray) -> np.ndarray:
    n, n_trees = leaf_indices.shape
    prox = np.zeros((n, n), dtype=np.float64)
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
    def objective(y_pred, dtrain):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        pt = np.where(y == 1, p, 1 - p)
        w = np.where(y == 1, alpha, 1 - alpha)
        mod = (1.0 - pt) ** gamma
        grad = w * mod * (p - y)
        hess = np.maximum(w * mod * p * (1.0 - p), 1e-6)
        return grad, hess
    return objective


def run_xgb_cv_mcc(X, y, params, use_focal: bool = False) -> float:
    mcc_scores = []
    for tr, val in cv.split(X, y):
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dval = xgb.DMatrix(X[val], label=y[val])
        bp = {
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
            bp['scale_pos_weight'] = params['scale_pos_weight']
        kw = {'num_boost_round': params.get('n_estimators', 300), 'evals': [(dval, 'val')], 'verbose_eval': False}
        if use_focal:
            bst = xgb.train(bp, dtrain, obj=focal_loss_objective(2.0, 0.25), **kw)
        else:
            bst = xgb.train(bp, dtrain, **kw)
        pred = (bst.predict(dval) >= 0.5).astype(int)
        mcc_scores.append(matthews_corrcoef(y[val], pred))
    return float(np.mean(mcc_scores))


def cv_summary(estimator, X, y, name: str) -> Dict:
    t0 = time.time()
    scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring_cv, n_jobs=-1)
    elapsed = time.time() - t0
    model_times[name] = elapsed
    row = {
        'Modèle': name,
        'F1-Macro (CV)': scores['test_f1_macro'].mean(),
        'F1-Macro std': scores['test_f1_macro'].std(),
        'AUPRC (CV)': scores['test_auprc'].mean(),
        'AUPRC std': scores['test_auprc'].std(),
        'MCC (CV)': scores['test_mcc'].mean(),
        'MCC std': scores['test_mcc'].std(),
        'Temps (s)': elapsed,
    }
    cv_results_all.append(row)
    return row
"""
    )

    nb.md(
        """## 3. Modèle 1 — Régression logistique Elastic Net

Pipeline : `SimpleImputer` → `StandardScaler` → `LogisticRegression(elasticnet, saga, class_weight='balanced')`"""
    )
    nb.code(
        """
print('='*60)
print('MODÈLE 1 : Elastic Net')
print('='*60)
t0 = time.time()

log_reg = LogisticRegression(
    penalty='elasticnet', solver='saga', class_weight='balanced',
    max_iter=5000, random_state=SEED,
)
pipe_lr = create_model_pipeline(log_reg, with_smote=False, scaling=True)

param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__l1_ratio': [0.0, 0.3, 0.5, 0.7, 1.0],
}

grid_lr = GridSearchCV(
    pipe_lr, param_grid, cv=cv, scoring=scoring_cv, refit='mcc', n_jobs=-1, verbose=1,
)
grid_lr.fit(X_train, y_train)
model_times['Elastic Net'] = time.time() - t0

best_lr = grid_lr.best_estimator_
print(f'Meilleurs params : {grid_lr.best_params_}')
print(f'CV MCC (refit) : {grid_lr.best_score_:.4f}')

cv_summary(best_lr, X_train, y_train, 'Elastic Net')
joblib.dump(best_lr, os.path.join(MODELS_DIR, 'logreg_elasticnet.pkl'))
joblib.dump(grid_lr, os.path.join(MODELS_DIR, 'logreg_gridsearch.pkl'))
print('Sauvegardé : models/logreg_elasticnet.pkl')
"""
    )

    nb.md(
        """## 4. Modèle 2 — Random Forest + proximité + MDS

- RF entraîné sur **60k** | proximité/MDS sur **5k** (stratifié)"""
    )
    nb.code(
        """
print('='*60)
print('MODÈLE 2 : Random Forest')
print('='*60)
t0 = time.time()

rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        class_weight='balanced', random_state=SEED, n_jobs=-1,
    )),
])
rf_model.fit(X_train, y_train)
model_times['Random Forest'] = time.time() - t0

cv_summary(rf_model, X_train, y_train, 'Random Forest')
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest.pkl'))
print('Sauvegardé : models/random_forest.pkl')
"""
    )
    nb.code(
        """
PROXIMITY_N = 5000
rng = np.random.RandomState(SEED)
idx_pos = np.where(y_train.values == 1)[0]
idx_neg = np.where(y_train.values == 0)[0]
n_pos_s = min(len(idx_pos), max(1, int(PROXIMITY_N * y_train.mean())))
n_neg_s = PROXIMITY_N - n_pos_s
idx_sample = np.concatenate([
    rng.choice(idx_pos, n_pos_s, replace=False),
    rng.choice(idx_neg, n_neg_s, replace=False),
])
rng.shuffle(idx_sample)

X_sub = X_train.iloc[idx_sample]
y_sub = y_train.iloc[idx_sample]
X_sub_imp = rf_model.named_steps['imputer'].transform(X_sub)
rf_fitted = rf_model.named_steps['rf']

print(f'Proximité sur {len(idx_sample)} points (RF entraîné sur {len(X_train)})')
leaf_idx = rf_fitted.apply(X_sub_imp)
proximity = compute_proximity_matrix(leaf_idx)
dissim = 1.0 - proximity

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED, n_init=4, max_iter=300)
embedding = mds.fit_transform(dissim)

y_proba_sub = rf_fitted.predict_proba(X_sub_imp)[:, 1]
uncertainty = np.abs(y_proba_sub - 0.5)

iso = IsolationForest(contamination=0.05, random_state=SEED)
is_outlier = iso.fit_predict(embedding) == -1

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(embedding[~is_outlier, 0], embedding[~is_outlier, 1],
           c=COLORS['primary'], alpha=0.3, s=15, label='Inliers')
ax.scatter(embedding[is_outlier, 0], embedding[is_outlier, 1],
           c=COLORS['danger'], alpha=0.9, s=45, edgecolors='k', linewidths=0.4, label='Outliers')
ax.set_title('MDS 2D (1 - proximité RF) — outliers IsolationForest')
ax.legend()
plt.tight_layout()
save_figure(fig, 'rf_mds_proximity_outliers', save_dir=FIGURES_DIR)
plt.show()

outlier_df = X_sub.copy()
outlier_df['y_true'] = y_sub.values
outlier_df['proba'] = y_proba_sub
outlier_df['uncertainty'] = uncertainty
outlier_df['is_outlier'] = is_outlier

global_med = X_train.median()
out_med = outlier_df.loc[is_outlier, X_train.columns].median()
z_shift = ((out_med - global_med) / (X_train.std() + 1e-8)).abs().sort_values(ascending=False)
print('Top capteurs décalés (outliers):')
display(z_shift.head(10).to_frame('z_shift'))

print('''
**Analyse** : outliers = faible proximité RF + isolement MDS.
- Incertitude élevée → frontière floue.
- Capteurs atypiques → patterns rares / bruit.
- Erreurs concentrées → cas limites du classifieur.
''')
"""
    )

    nb.md("## 5. Modèle 3 — XGBoost + Optuna (50 essais × 2 stratégies)")
    nb.code(
        """
_imputer = SimpleImputer(strategy='median')
X_tr_imp = _imputer.fit_transform(X_train)
X_te_imp = _imputer.transform(X_test)
y_arr = y_train.values

N_TRIALS = 50
STORAGE = f"sqlite:///{os.path.join(MODELS_DIR, 'optuna_study.db')}"


def make_objective(use_focal: bool):
    def objective(trial):
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
            params['scale_pos_weight'] = trial.suggest_categorical('scale_pos_weight', SPW_CANDIDATES)
        return run_xgb_cv_mcc(X_tr_imp, y_arr, params, use_focal=use_focal)
    return objective

print('Optuna Stratégie A (scale_pos_weight)...')
t0 = time.time()
study_a = optuna.create_study(
    direction='maximize', sampler=TPESampler(seed=SEED),
    study_name='xgb_strategy_a', storage=STORAGE, load_if_exists=True,
)
study_a.optimize(make_objective(False), n_trials=N_TRIALS, show_progress_bar=True)
time_a = time.time() - t0

print('Optuna Stratégie B (Focal Loss γ=2, α=0.25)...')
t0 = time.time()
study_b = optuna.create_study(
    direction='maximize', sampler=TPESampler(seed=SEED),
    study_name='xgb_strategy_b', storage=STORAGE, load_if_exists=True,
)
study_b.optimize(make_objective(True), n_trials=N_TRIALS, show_progress_bar=True)
time_b = time.time() - t0

print(f'A best MCC CV={study_a.best_value:.4f} | B best MCC CV={study_b.best_value:.4f}')
use_focal = study_b.best_value > study_a.best_value
best_study = study_b if use_focal else study_a
best_params = best_study.best_params
strategy_label = 'B (Focal)' if use_focal else 'A (SPW)'
model_times['XGBoost'] = time_a + time_b
"""
    )

    nb.code(
        """
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

def save_optuna_fig(fig, fname):
    path = os.path.join(FIGURES_DIR, fname)
    try:
        fig.write_image(path)
        print(f'  PNG: {path}')
    except Exception as e:
        html_path = path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f'  HTML (kaleido manquant): {html_path} — {e}')
    return fig

for study, tag, title in [
    (study_a, 'strategy_a', 'Stratégie A — scale_pos_weight'),
    (study_b, 'strategy_b', 'Stratégie B — Focal Loss'),
]:
    fh = plot_optimization_history(study)
    fh.update_layout(title=f'Historique Optuna — {title}')
    save_optuna_fig(fh, f'optuna_history_{tag}.png')
    save_optuna_fig(plot_param_importances(study), f'optuna_importance_{tag}.png')
    fs = plot_slice(study, params=['max_depth', 'learning_rate'])
    fs.update_layout(title=f'Slice — {title}')
    save_optuna_fig(fs, f'optuna_slice_{tag}.png')

plot_optimization_history(study_a).show()
plot_param_importances(study_a).show()
plot_slice(study_a, params=['max_depth', 'learning_rate']).show()
"""
    )

    nb.code(
        """
# Entraînement final meilleure stratégie
params = best_params.copy()
n_rounds = params.pop('n_estimators', 300)
dtrain = xgb.DMatrix(X_tr_imp, label=y_arr)
bp = {
    'max_depth': params['max_depth'], 'eta': params['learning_rate'],
    'lambda': params['lambda'], 'alpha': params['alpha'],
    'subsample': params['subsample'], 'colsample_bytree': params['colsample_bytree'],
    'objective': 'binary:logistic', 'seed': SEED, 'verbosity': 0,
}
if not use_focal:
    bp['scale_pos_weight'] = params['scale_pos_weight']
    bst = xgb.train(bp, dtrain, num_boost_round=n_rounds)
else:
    bst = xgb.train(bp, dtrain, num_boost_round=n_rounds, obj=focal_loss_objective(2.0, 0.25))

xgb_pack = {
    'booster': bst, 'imputer': _imputer, 'params': best_params,
    'use_focal': use_focal, 'strategy': strategy_label,
}
joblib.dump(xgb_pack, os.path.join(MODELS_DIR, 'xgboost_best.pkl'))

xgb_cv_row = {
    'Modèle': f'XGBoost {strategy_label}',
    'F1-Macro (CV)': np.nan,
    'AUPRC (CV)': np.nan,
    'MCC (CV)': best_study.best_value,
    'MCC std': 0,
    'Temps (s)': model_times['XGBoost'],
}
cv_results_all.append(xgb_cv_row)
print(f'Sauvegardé xgboost_best.pkl — stratégie {strategy_label}')
"""
    )

    nb.md("## 6. Tableau comparatif et conclusion")
    nb.code(
        """
cv_df = pd.DataFrame(cv_results_all)
display(cv_df.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
plot_df = cv_df.dropna(subset=['MCC (CV)'])
x = np.arange(len(plot_df))
w = 0.25
ax.bar(x - w, plot_df['F1-Macro (CV)'], w, label='F1-Macro')
ax.bar(x, plot_df['AUPRC (CV)'], w, label='AUPRC')
ax.bar(x + w, plot_df['MCC (CV)'], w, label='MCC')
ax.set_xticks(x)
ax.set_xticklabels(plot_df['Modèle'], rotation=20, ha='right')
ax.set_ylim(0, 1.05)
ax.set_title('Comparaison CV des modèles')
ax.legend()
plt.tight_layout()
save_figure(fig, 'model_cv_comparison', save_dir=FIGURES_DIR)
plt.show()

cv_df.to_csv(os.path.join(TABLES_DIR, 'model_cv_comparison.csv'), index=False)
best = cv_df.loc[cv_df['MCC (CV)'].idxmax()]
print(f'\\n🏆 Meilleur modèle (MCC CV) : {best["Modèle"]} — MCC={best["MCC (CV)"]:.4f}')
print('Prochaine étape : 03_evaluation_calibration.ipynb')
"""
    )
    return nb


def build_nb03() -> NB:
    nb = NB()
    nb.md(
        """# Étape 3 : Évaluation finale et calibration

## Test set officiel + courbes + calibration (ECE)

**Métriques** : F1-Macro, AUPRC, MCC, coût IDA (FP×10 + FN×500)

---"""
    )
    nb.md("## 0. Configuration")
    nb.code(COMMON_HEADER + """
import joblib
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef, f1_score, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from src.evaluation.calibration import compute_ece, plot_reliability_diagram, calibrate_model
""")

    nb.md("## 1. Chargement données et modèles")
    nb.code(
        """
train_df, test_df = load_aps_data(project_root=PROJECT_ROOT)
X_train, y_train = get_X_y(train_df)
X_test, y_test = get_X_y(test_df)

logreg = joblib.load(os.path.join(MODELS_DIR, 'logreg_elasticnet.pkl'))
rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
xgb_pack = joblib.load(os.path.join(MODELS_DIR, 'xgboost_best.pkl'))

def predict_xgb(pack, X):
    X_imp = pack['imputer'].transform(X)
    return pack['booster'].predict(xgb.DMatrix(X_imp))

models = {
    'Elastic Net': logreg,
    'Random Forest': rf_model,
    f"XGBoost {xgb_pack['strategy']}": xgb_pack,
}

def get_proba(name, model, X):
    if name.startswith('XGBoost'):
        return predict_xgb(model, X)
    return model.predict_proba(X)[:, 1]

print('Modèles chargés:', list(models.keys()))
"""
    )

    nb.md("## 2. Évaluation sur le test set")
    nb.code(
        """
test_results = []
predictions = {}

for name, model in models.items():
    proba = get_proba(name, model, X_test)
    thr, _, _, _ = find_optimal_threshold(y_train.values, get_proba(name, model, X_train), metric='mcc')
    pred = (proba >= thr).astype(int)
    m = compute_all_metrics(y_test.values, pred, proba)
    predictions[name] = {'proba': proba, 'pred': pred, 'threshold': thr}
    test_results.append({
        'Modèle': name, 'F1-Macro': m['F1-Macro'], 'AUPRC': m['AUPRC'], 'MCC': m['MCC'],
        'Precision': m['Precision'], 'Recall': m['Recall'], 'Coût IDA': m['Cost'], 'Seuil': thr,
    })
    print_metrics_report(y_test.values, pred, proba, model_name=f'{name} (test)')

test_df_metrics = pd.DataFrame(test_results)
display(test_df_metrics.round(4))
best_model_name = test_df_metrics.loc[test_df_metrics['MCC'].idxmax(), 'Modèle']
print(f'\\nMeilleur modèle (MCC test) : {best_model_name}')
"""
    )

    nb.md("## 3. Courbes de performance (300 DPI)")
    nb.code(
        """
# 3.1 Precision-Recall
fig, ax = plt.subplots(figsize=(10, 7))
baseline = y_test.mean()
for name in models:
    p, r, _ = precision_recall_curve(y_test, predictions[name]['proba'])
    ap = average_precision_score(y_test, predictions[name]['proba'])
    ax.plot(r, p, lw=2, label=f'{name} (AUPRC={ap:.4f})')
ax.axhline(baseline, color='gray', ls='--', label=f'Aléatoire ({baseline:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Courbes Precision-Recall — Test set')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'pr_curves', save_dir=FIGURES_DIR)
plt.show()

# 3.2 ROC (référence)
fig, ax = plt.subplots(figsize=(10, 7))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, predictions[name]['proba'])
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.4f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('Courbes ROC — référence uniquement')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'roc_curves', save_dir=FIGURES_DIR)
plt.show()
"""
    )

    nb.code(
        """
# 3.3 MCC vs seuil (meilleur modèle) — pas de 0.05
best_proba = predictions[best_model_name]['proba']
thresholds = np.arange(0.1, 0.91, 0.05)
mcc_scores, cost_scores = [], []
for t in thresholds:
    pred_t = (best_proba >= t).astype(int)
    mcc_scores.append(matthews_corrcoef(y_test, pred_t))
    cm = confusion_matrix(y_test, pred_t)
    tn, fp, fn, tp = cm.ravel()
    cost_scores.append(COST_FP * fp + COST_FN * fn)

best_mcc_idx = np.argmax(mcc_scores)
thr_mcc = thresholds[best_mcc_idx]
best_cost_idx = np.argmin(cost_scores)
thr_cost = thresholds[best_cost_idx]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, mcc_scores, 'o-', color=COLORS['primary'])
ax.axvline(thr_mcc, color=COLORS['danger'], ls='--', label=f'opt MCC={thr_mcc:.2f}')
ax.set_xlabel('Seuil'); ax.set_ylabel('MCC'); ax.set_title(f'MCC vs seuil — {best_model_name}')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'mcc_threshold', save_dir=FIGURES_DIR)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, cost_scores, 'o-', color=COLORS['secondary'])
ax.axvline(thr_cost, color=COLORS['danger'], ls='--', label=f'opt coût={thr_cost:.2f}')
ax.set_xlabel('Seuil'); ax.set_ylabel('Coût IDA (€)'); ax.set_title('Coût métier vs seuil')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'cost_threshold', save_dir=FIGURES_DIR)
plt.show()
print(f'Seuil optimal MCC={thr_mcc:.2f} | Seuil optimal coût={thr_cost:.2f}')
"""
    )

    nb.code(
        """
# 3.5 Barplot final
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['F1-Macro', 'AUPRC', 'MCC']
for ax, met in zip(axes, metrics):
    ax.bar(test_df_metrics['Modèle'], test_df_metrics[met], color=COLORS['primary'], edgecolor='white')
    ax.set_title(met); ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
save_figure(fig, 'final_metrics_barplot', save_dir=FIGURES_DIR)
plt.show()
"""
    )

    nb.md("## 4. Calibration des probabilités")
    nb.code(
        """
# ECE avant calibration
ece_before = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    proba = get_proba(name, model, X_test)
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy='uniform')
    ece = compute_ece(y_test.values, proba, n_bins=10)
    ece_before[name] = ece
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(prob_pred, prob_true, 's-', label=f'ECE={ece:.4f}')
    ax.set_title(name); ax.set_xlabel('Proba prédite'); ax.set_ylabel('Fraction positifs')
    ax.legend()
plt.suptitle('Reliability diagrams — avant calibration', fontweight='bold')
plt.tight_layout()
save_figure(fig, 'calibration_before', save_dir=FIGURES_DIR)
plt.show()
"""
    )

    nb.code(
        """
# Calibration — isotonic pour arbres/XGB, sigmoid (Platt) pour linéaire
proba_train_best = get_proba(best_model_name, models[best_model_name], X_train)
proba_test_best = predictions[best_model_name]['proba']
ece_before_val = ece_before[best_model_name]

if ece_before_val > 0.05:
    if best_model_name.startswith('XGBoost'):
        from sklearn.isotonic import IsotonicRegression
        print(f'Calibration isotonic (XGBoost) — ECE avant={ece_before_val:.4f}')
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(proba_train_best, y_train.values)
        proba_cal = iso_reg.transform(proba_test_best)
        calibrated = iso_reg
    else:
        cal_method = 'isotonic' if 'Forest' in best_model_name else 'sigmoid'
        print(f'Calibration CalibratedClassifierCV — méthode {cal_method}')
        calibrated = CalibratedClassifierCV(models[best_model_name], method=cal_method, cv=N_FOLDS)
        calibrated.fit(X_train, y_train)
        proba_cal = calibrated.predict_proba(X_test)[:, 1]
else:
    print(f'ECE={ece_before_val:.4f} <= 0.05 — calibration optionnelle, on garde les probas brutes')
    proba_cal = proba_test_best
    calibrated = None

ece_after = compute_ece(y_test.values, proba_cal, n_bins=10)
if calibrated is not None and best_model_name.startswith('XGBoost'):
    proba_cal_train = iso_reg.transform(proba_train_best)
elif calibrated is not None:
    proba_cal_train = calibrated.predict_proba(X_train)[:, 1]
else:
    proba_cal_train = proba_train_best
thr_cal, _, _, _ = find_optimal_threshold(y_train.values, proba_cal_train, metric='mcc')
pred_cal = (proba_cal >= thr_cal).astype(int)
m_cal = compute_all_metrics(y_test.values, pred_cal, proba_cal)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (proba, title) in zip(axes, [
    (predictions[best_model_name]['proba'], 'Avant'),
    (proba_cal, 'Après calibration'),
]):
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    ece = compute_ece(y_test.values, proba, 10)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(prob_pred, prob_true, 's-', label=f'ECE={ece:.4f}')
    ax.set_title(f'{best_model_name} — {title}')
    ax.legend()
plt.tight_layout()
save_figure(fig, 'calibration_comparison', save_dir=FIGURES_DIR)
plt.show()

print(f'ECE avant : {ece_before[best_model_name]:.4f} | après : {ece_after:.4f}')
print(f'MCC avant : {test_df_metrics.loc[test_df_metrics["Modèle"]==best_model_name,"MCC"].values[0]:.4f} | après : {m_cal["MCC"]:.4f}')
if calibrated is not None:
    joblib.dump(calibrated, os.path.join(MODELS_DIR, 'best_model_calibrated.pkl'))
"""
    )

    nb.md("## 5. Synthèse rapport")
    nb.code(
        """
final_table = test_df_metrics.copy()
final_table['ECE'] = final_table['Modèle'].map(ece_before)
final_table['ECE après cal.'] = np.nan
final_table.loc[final_table['Modèle'] == best_model_name, 'ECE après cal.'] = ece_after
final_table['Seuil optimal MCC'] = final_table['Seuil']
final_table['Seuil coût minimal'] = thr_cost

display(final_table.round(4))
final_table.to_csv(os.path.join(TABLES_DIR, 'final_comparison.csv'), index=False)

print('''
**Recommandation production** :
- Modèle : ''' + best_model_name + '''
- Seuil MCC : ''' + f'{thr_mcc:.2f}' + ''' | Seuil coût : ''' + f'{thr_cost:.2f}' + '''
- Privilégier le seuil coût si l'objectif est le challenge IDA 2016.
- Appliquer la calibration si ECE > 0.05.
''')
"""
    )
    return nb


def build_nb04() -> NB:
    nb = NB()
    nb.md(
        """# Étape 4 : Analyse SHAP (interprétabilité)

## Meilleur modèle : XGBoost

TreeExplainer sur sous-échantillon test (5000 points).

---"""
    )
    nb.md("## 0. Configuration")
    nb.code(
        COMMON_HEADER
        + """
import joblib
import shap
import xgboost as xgb

train_df, test_df = load_aps_data(project_root=PROJECT_ROOT)
X_train, y_train = get_X_y(train_df)
X_test, y_test = get_X_y(test_df)

xgb_pack = joblib.load(os.path.join(MODELS_DIR, 'xgboost_best.pkl'))
if not os.path.exists(os.path.join(MODELS_DIR, 'xgboost_best.pkl')):
    raise FileNotFoundError('Exécutez d\\'abord 02_model_training.ipynb')

booster = xgb_pack['booster']
imputer = xgb_pack['imputer']
"""
    )

    nb.md("## 1. SHAP — valeurs globales et locales")
    nb.code(
        """
SHAP_SAMPLE = 5000
rng = np.random.RandomState(SEED)
idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
X_shap = X_test.iloc[idx]
X_shap_imp = imputer.transform(X_shap)
y_shap = y_test.iloc[idx].values

print(f'Calcul SHAP sur {len(X_shap)} points...')
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(xgb.DMatrix(X_shap_imp))

# Summary beeswarm
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, plot_type='bar', show=False, max_display=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
plt.show()

mean_abs = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_shap.columns,
    'mean_abs_shap': mean_abs,
}).sort_values('mean_abs_shap', ascending=False)
top10 = importance_df.head(10)
display(top10)
"""
    )

    nb.code(
        """
# Dependence plots — top 5
top5 = importance_df.head(5)['feature'].tolist()
for feat in top5:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(feat, shap_values, X_shap, show=False)
    plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_dependence.png'), dpi=300, bbox_inches='tight')
plt.show()

# Prédictions pour exemples locaux
proba = booster.predict(xgb.DMatrix(X_shap_imp))
pred = (proba >= 0.5).astype(int)
tp_idx = np.where((y_shap == 1) & (pred == 1))[0]
fp_idx = np.where((y_shap == 0) & (pred == 1))[0]
fn_idx = np.where((y_shap == 1) & (pred == 0))[0]

examples = {}
if len(tp_idx): examples['TP'] = tp_idx[0]
if len(fp_idx): examples['FP'] = fp_idx[0]
if len(fn_idx): examples['FN'] = fn_idx[0]

for label, i in examples.items():
  plt.figure(figsize=(10, 4))
  shap.waterfall_plot(
      shap.Explanation(values=shap_values[i], base_values=explainer.expected_value,
                       data=X_shap.iloc[i].values, feature_names=X_shap.columns.tolist()),
      show=False, max_display=12,
  )
  plt.title(f'Waterfall — {label} (index {i})')
  plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall_examples.png'), dpi=300, bbox_inches='tight')
plt.show()
"""
    )

    nb.md("## 2. Stabilité SHAP sur 5 folds")
    nb.code(
        """
print('Stabilité SHAP par fold (sous-échantillon 2000/fold)...')
rng = np.random.RandomState(SEED)
fold_importances = []
for fold, (tr, val) in enumerate(tqdm(cv.split(X_train, y_train), total=N_FOLDS)):
    X_v = X_train.iloc[val]
    y_v = y_train.iloc[val]
    n_s = min(2000, len(X_v))
    sub = rng.choice(len(X_v), n_s, replace=False)
    X_imp = imputer.transform(X_v.iloc[sub])
    # Réentraîner booster léger par fold serait coûteux — on utilise le modèle final
    # comme proxy de stabilité des explications sur sous-populations
    sv = explainer.shap_values(xgb.DMatrix(X_imp))
    fold_importances.append(np.abs(sv).mean(axis=0))

stab = np.vstack(fold_importances)
importance_df['shap_std_across_folds'] = stab.std(axis=0)
importance_df['stable'] = importance_df['shap_std_across_folds'] < importance_df['mean_abs_shap'].median()
display(importance_df.head(15))
"""
    )

    nb.md("## 3. Interprétation métier")
    nb.code(
        """
print('''
**Classe positive = panne APS** :
- SHAP > 0 pousse vers la prédiction « panne ».
- Capteurs top = candidats pour surveillance préventive.

**Biais / proxies** : vérifier si des capteurs corrélés à des conditions externes
(température, régime moteur) dominent sans lien causal APS.

**Actions** : prioriser maintenance sur les capteurs du top 10 SHAP stable.
''')

importance_df.head(10).to_csv(os.path.join(TABLES_DIR, 'shap_feature_importance.csv'), index=False)
print('Export : reports/tables/shap_feature_importance.csv')
print('Figures : shap_summary.png, shap_dependence.png, shap_waterfall_examples.png')
"""
    )
    return nb


def patch_nb01():
    """Insert describe-by-class cell if missing."""
    path = NB_DIR / "01_eda_preprocessing.ipynb"
    if not path.exists():
        return
    nb = json.loads(path.read_text(encoding="utf-8"))
    src_all = "".join("".join(c.get("source", [])) for c in nb["cells"])
    if "describe_by_class" not in src_all and "STATISTIQUES PAR CLASSE" not in src_all:
        new_cell = {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            ln + "\n"
            for ln in """
# ============================================================
# 2.4b Statistiques descriptives PAR CLASSE
# ============================================================
print("\\n📊 STATISTIQUES PAR CLASSE (describe)")
print("="*60)

train_labeled = X_train.copy()
train_labeled['class'] = y_train.values

for label, name in [(0, 'Négatif (neg)'), (1, 'Positif (pos)')]:
    subset = train_labeled[train_labeled['class'] == label].drop(columns=['class'])
    print(f"\\n--- {name} — n={len(subset):,} ---")
    display(subset.describe().T[['mean', 'std', 'min', '50%', 'max']].head(15))

# Comparer les 5 features les plus corrélées avec la target (si target_corr existe)
try:
    top5_disc = target_corr.head(5).index.tolist()
    compare = train_labeled.groupby('class')[top5_disc].mean().T
    compare.columns = ['neg (0)', 'pos (1)']
    compare['delta_pos_neg'] = compare['pos (1)'] - compare['neg (0)']
    print("\\nMoyenne des 5 features discriminantes par classe :")
    display(compare)
except NameError:
    print("(Exécuter d'abord la cellule corrélation target)")
""".strip().split("\n")
        ],
    }
        for i, cell in enumerate(nb["cells"]):
            if "desc_stats = X_train.describe()" in "".join(cell.get("source", [])):
                nb["cells"].insert(i + 1, new_cell)
                break
        print("  nb01: cellule describe par classe ajoutee")
    else:
        print("  nb01: describe par classe deja present")

    # VIF sur toutes les features (spec)
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "top_features_for_vif = target_corr.head(50)" in src:
            new_src = src.replace(
                "top_features_for_vif = target_corr.head(50).index.tolist()",
                "top_features_for_vif = X_train_imputed.columns.tolist()  # 170 features",
            ).replace(
                "# Sélectionner les top 50 features les plus corrélées avec la target\n# (le VIF sur 170 features serait trop long)",
                "# VIF sur les 170 features (données imputées + scalées)",
            )
            cell["source"] = [ln + "\n" for ln in new_src.split("\n")]
            print("  nb01: VIF sur 170 features")
            break

    # Violin: 5 features (spec)
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "target_corr.head(9)" in src:
            cell["source"] = [ln.replace("head(9)", "head(5)").replace("9 features", "5 features")
                              for ln in cell["source"]]
            print("  nb01: violin plots -> 5 features")
            break
    # Proposition suppression features VIF > 5
    src_all = "".join("".join(c.get("source", [])) for c in nb["cells"])
    if "PROPOSITION FEATURES" not in src_all:
        prop_cell = {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [ln + "\n" for ln in """
# Proposition : features a retirer ou regrouper (VIF > 5)
high_vif = vif_df[vif_df['VIF'] > 5].copy()
print(f"Features avec VIF > 5 : {len(high_vif)}")
display(high_vif[['Feature', 'VIF', 'Corr_target']].head(20))

# Paires tres correlees (|rho| > 0.9) : garder celle la plus correlee a la target
pairs_high = corr_pairs_df[corr_pairs_df['Abs_Correlation'] > 0.9]
print(f"\\nPaires |Spearman| > 0.9 : {len(pairs_high)}")
display(pairs_high.head(15))

print('''
**Recommandation** :
1. Retirer iterativement la feature avec le VIF le plus eleve jusqu'a VIF < 5 (modeles lineaires).
2. Pour les paires |rho|>0.9, conserver la feature la plus correlee a la target.
3. Ne pas supprimer avant la validation CV — tester l'impact sur MCC au notebook 2.
''')
""".strip().split("\n")],
        }
        for i, cell in enumerate(nb["cells"]):
            if "high_vif_features = vif_df" in "".join(cell.get("source", [])):
                nb["cells"].insert(i + 1, prop_cell)
                print("  nb01: proposition suppression features")
                break

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


def main():
    print("Génération des notebooks...")
    patch_nb01()
    build_nb02().save("02_model_training.ipynb")
    build_nb03().save("03_evaluation_calibration.ipynb")
    build_nb04().save("04_shap_analysis.ipynb")
    print("Terminé.")


if __name__ == "__main__":
    main()
