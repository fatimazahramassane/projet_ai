---
title: "Classification Robuste et Analyse de Décision en Environnement Critique"
subtitle: "Prédiction de pannes APS — Scania Trucks"
author: "[Nom Prénom]"
date: "Juin 2026"
lang: fr
toc: true
toc-depth: 3
numbersections: true
geometry: margin=2.5cm
fontsize: 11pt
documentclass: article
---

\newpage

# Page de garde

**Titre :** Classification Robuste et Analyse de Décision en Environnement Critique

**Sous-titre :** Prédiction de pannes du système de pression d'air (APS) — Scania Trucks

**Méthodologie :** Feature engineering exploratoire, Elastic Net, Random Forest (proximité OOB), XGBoost cost-sensitive (Optuna), calibration des probabilités, interprétabilité SHAP

**Dataset :** UCI *APS Failure and Operational Data at Scania Trucks* — 60 000 observations d'entraînement, 170 capteurs, déséquilibre ≈ 59:1 ({{pos_pct}} % de positifs)

**Encadrement :** [Nom du professeur / UE]

**Auteur :** [Nom Prénom]

**Date :** Juin 2026

\newpage

# Table des matières

*Générée automatiquement à l'export PDF (Pandoc/LaTeX).*

\newpage

# Introduction

## Contexte métier

Les camions Scania embarquent un vaste réseau de capteurs supervisant notamment le **système de pression d'air (APS)**, qui alimente le freinage et les changements de vitesse. Une défaillance APS non détectée peut provoquer une immobilisation coûteuse ou un risque sécuritaire. Le jeu de données **APS Failure at Scania Trucks** (Scania CV AB, challenge industriel IDA 2016) propose de prédire, à partir de lectures de capteurs anonymisées, si un enregistrement correspond à une **panne liée à l'APS** (classe positive) ou à une autre défaillance (classe négative).

## Problématique

Trois difficultés structurent ce problème :

1. **Déséquilibre extrême** — environ 59 000 négatifs pour 1 000 positifs sur l'ensemble d'entraînement (ratio ≈ 59:1, soit {{pos_pct}} % de positifs).
2. **Coût asymétrique de classification** (métrique officielle du challenge) :
   - **Faux positif (FP)** : 10 € — contrôle atelier inutile ;
   - **Faux négatif (FN)** : 500 € — panne non anticipée.
3. **Données bruitées et incomplètes** — valeurs `na`, capteurs redondants (multicolinéarité), distributions non gaussiennes.

Un modèle naïf prédisant toujours « pas de panne APS » atteindrait une accuracy ≈ 98,3 % tout en étant **inacceptable** opérationnellement (FN massifs).

## Objectifs

| Objectif | Approche retenue |
|----------|------------------|
| Robustesse au déséquilibre | `class_weight`, SMOTE dans `imblearn.Pipeline`, `scale_pos_weight` / focal loss (XGBoost) |
| Décision fiable | Calibration (Platt / isotonic), analyse ECE, seuil optimisé sur coût métier |
| Interprétabilité | SHAP `TreeExplainer` sur le meilleur modèle arborescent |
| Reproductibilité | `seed=42`, `StratifiedKFold(5)`, code versionné (`src/`, notebooks 01–04) |

## Structure du rapport

- **Chapitre 1** — Analyse exploratoire et préparation (notebook 01).
- **Chapitre 2** — Développement de trois modèles (notebook 02).
- **Chapitre 3** — Évaluation sur le test set et calibration (notebook 03).
- **Chapitre 4** — Interprétabilité SHAP (notebook 04).
- **Conclusion** — Synthèse, limites, perspectives et recommandations de déploiement.

> **Mise à jour automatique des chiffres :** exécuter `python scripts/extract_report_metrics.py --fill reports/rapport_aps_scania.md` après les notebooks pour remplir les balises `{{...}}` à partir de `data/processed/eda_results.joblib` et `reports/tables/*.csv`.

\newpage

# Chapitre 1 — Analyse exploratoire et préparation (Étape 1)

## 1.1 Statistiques descriptives

### Dimensions et cible

| Indicateur | Valeur |
|------------|--------|
| Train | {{train_shape}} (attendu : 60 000 × 171) |
| Test | {{test_shape}} (attendu : 16 000 × 171) |
| Features | 170 capteurs + `class` |
| Positifs (panne APS) | {{n_pos}} |
| Négatifs | {{n_neg}} |
| Proportion positifs | {{pos_pct}} % |
| `scale_pos_weight` théorique (XGBoost) | ≈ {{scale_pos_weight}} |

Le chargement gère l'en-tête GPL (~20 lignes), convertit `na` en `NaN` et encode `pos`→1, `neg`→0 (`src/data/loader.py`).

### Valeurs manquantes

Le taux moyen de NaN par feature est d'environ **{{mean_missing_pct}} %** ; **{{features_with_missing_gt50pct}}** features dépassent 50 % de valeurs manquantes. L'imputation **médiane** est retenue dans les pipelines de modélisation (robuste aux outliers), appliquée **uniquement sur les folds d'entraînement** pour éviter toute fuite.

![Distribution de la classe cible](../reports/figures/01_class_distribution.png){ width=90% }

*Figure 1.1 — Distribution des classes sur l'ensemble d'entraînement.*

![Taux de valeurs manquantes (top 30 features)](../reports/figures/02_missing_values.png){ width=90% }

*Figure 1.2 — Analyse des valeurs manquantes.*

## 1.2 Analyse de colinéarité

### Corrélation de Spearman

Nous utilisons la corrélation de **Spearman** (rangs) plutôt que Pearson car :

- les relations capteur–capteur sont souvent **monotones mais non linéaires** ;
- les distributions sont **asymétriques** avec outliers (données industrielles).

La heatmap porte sur les 20 features les plus corrélées entre elles (sélection par paires |ρ| élevées).

![Heatmap Spearman — top 20 features](../reports/figures/04_correlation_heatmap.png){ width=85% }

*Figure 1.3 — Matrice de corrélation de Spearman.*

![Corrélation Spearman avec la cible (top 20)](../reports/figures/05_target_correlation.png){ width=85% }

*Figure 1.4 — Features les plus corrélées à la cible.*

### Variance Inflation Factor (VIF)

Pour chaque feature \(X_j\), le VIF mesure l'inflation de variance due à la multicolinéarité :

\[
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}
\]

où \(R_j^2\) est le \(R^2\) de la régression de \(X_j\) sur les autres features.

| Interprétation | Seuil retenu |
|----------------|--------------|
| Pas de colinéarité | VIF = 1 |
| Modérée | 1 < VIF < 5 |
| Problématique | **VIF > 5** (James et al., 2013) |
| Sévère | VIF > 10 |

**Résultat :** {{vif_count_gt5}} features présentent un VIF > 5. Exemples (top) : {{vif_top10_features}}.

![Analyse VIF (top 30)](../reports/figures/06_vif_analysis.png){ width=90% }

*Figure 1.5 — VIF par feature.*

Le tableau détaillé est exporté dans `reports/tables/top_10_vif_scores.csv`.

**Décision modélisation :** ne pas supprimer agressivement avant validation croisée — les modèles L1 (Elastic Net) et arbres gèrent la redondance ; une réduction itérative du VIF reste une piste pour la régression logistique seule.

## 1.3 Visualisations avancées

### Violin plots

Les cinq features les plus corrélées à la cible (Spearman) sont visualisées par classe pour comparer **forme, dispersion et séparabilité**.

![Violin plots — top 5 features discriminantes](../reports/figures/07_violin_plots.png){ width=95% }

*Figure 1.6 — Distributions par classe.*

### Projection UMAP 2D

**UMAP** (McInnes et al., 2018) préserve mieux la structure globale que t-SNE, avec un coût calcul raisonnable. Paramètres : `n_neighbors=15`, `min_dist=0.1`, `metric='euclidean'`, `random_state=42`.

![Projection UMAP 2D](../reports/figures/08_umap_2d.png){ width=85% }

*Figure 1.7 — Structure des données en 2D.*

**Lecture :** les classes présentent un **chevauchement partiel** — la séparation n'est pas linéairement parfaite, ce qui justifie des modèles non linéaires (forêts, boosting) en plus d'une baseline Elastic Net.

## 1.4 Comparaison des stratégies anti-déséquilibre

**Protocole :** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, pipeline `SimpleImputer(median)` + `StandardScaler` + classificateur.

| Stratégie | Mécanisme | F1-Macro (CV) | MCC (CV) |
|-----------|-----------|---------------|----------|
| **A — Algorithme** | `LogisticRegression(class_weight='balanced')` | {{imbalance_A_f1_mean}} ± {{imbalance_A_f1_std}} | {{imbalance_A_mcc_mean}} ± {{imbalance_A_mcc_std}} |
| **B — Données** | `SMOTE` dans `imblearn.Pipeline` (train fold uniquement) | {{imbalance_B_f1_mean}} ± {{imbalance_B_f1_std}} | {{imbalance_B_mcc_mean}} ± {{imbalance_B_mcc_std}} |

![Comparaison des stratégies par fold](../reports/figures/09_imbalance_comparison.png){ width=95% }

*Figure 1.8 — F1-Macro et MCC par fold.*

### Conclusion Chapitre 1

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| **class_weight** | Rapide, pas de duplication synthétique, pas de sur-apprentissage sur points SMOTE | Peut sous-représenter la frontière de décision |
| **SMOTE** | Densifie la région des positifs, peut améliorer le rappel | Coût calcul, risque de bruit si voisinage mal choisi |

**Choix pour la suite :** les modèles linéaires et forêts utilisent **`class_weight='balanced'`** ; XGBoost exploite **`scale_pos_weight ≈ 59`** et une variante **focal loss** (stratégie B, notebook 02). La stratégie retenue pour la régression logistique de comparaison est celle qui maximise le **MCC en CV** (voir métriques ci-dessus).

**Anti-leakage validé :** dimensions train/test inchangées ; SMOTE confiné aux folds d'entraînement.

\newpage

# Chapitre 2 — Développement des modèles (Étape 2)

## 2.1 Modèle 1 — Régression logistique Elastic Net (baseline)

### Justification théorique

L'**Elastic Net** combine pénalités L1 (Lasso) et L2 (Ridge) :

\[
\min_{\beta} \; -\log L(\beta) + \lambda \left( \alpha \|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2 \right)
\]

- **L1** : sélection / sparsité des capteurs ;
- **L2** : stabilité face à la multicolinéarité (VIF élevés).

C'est une baseline **interprétable** et rapide.

### Hyperparamètres

| Paramètre | Grille | Rôle |
|-----------|--------|------|
| `C` | [0.001, 0.01, 0.1, 1, 10] | Inverse de la force de régularisation |
| `l1_ratio` | [0, 0.3, 0.5, 0.7, 1] | Part L1 dans la pénalité elastic net |

**Validation :** `GridSearchCV`, 5 folds stratifiés, `scoring='matthews_corrcoef'`, `refit='mcc'`, `solver='saga'`, `class_weight='balanced'`.

### Résultats (à compléter depuis notebook 02)

| Indicateur | Valeur |
|------------|--------|
| Meilleur `C` | *[ex. 0.1 — voir `grid_lr.best_params_`]* |
| Meilleur `l1_ratio` | *[ex. 0.5]* |
| MCC CV (refit) | *[voir sortie GridSearch]* |
| F1-Macro CV | *[voir `cv_summary`]* |
| AUPRC CV | *[voir `cv_summary`]* |
| Temps d'entraînement | *[secondes]* |

## 2.2 Modèle 2 — Random Forest + proximité OOB

### Entraînement

| Hyperparamètre | Valeur |
|----------------|--------|
| `n_estimators` | 500 |
| `max_depth` | `None` |
| `min_samples_leaf` | 5 |
| `class_weight` | `'balanced'` |
| `random_state` | 42 |

### Matrice de proximité et outliers

La proximité entre observations \(i\) et \(j\) est la fréquence où elles partagent la même feuille terminale. Une matrice 60 000×60 000 est irréalisable (~28 Go) ; nous sous-échantillonnons **5 000** points (**stratifiés**).

- **MDS** sur la dissimilarité \(1 - \text{proximité}\) → visualisation 2D ;
- **Isolation Forest** (`contamination=0.05`) pour repérer les points isolés.

![MDS proximité RF + outliers](../reports/figures/rf_mds_proximity_outliers.png){ width=90% }

*Figure 2.1 — Structure de proximité et outliers (~5 %).*

**Interprétation :**

- **Nombre d'outliers détectés :** ≈ 250 sur 5 000 (5 %).
- Ces points se situent souvent en **zone de frontière** ou présentent des combinaisons de capteurs **atypiques**.
- Le modèle « hésite » lorsque les vecteurs s'éloignent des régions denses apprises — interactions non linéaires et bruit capteur.

### Résultats CV / test

| Métrique | Valeur |
|----------|--------|
| F1-Macro | *[CV / test]* |
| AUPRC | *[CV / test]* |
| MCC | *[CV / test]* |

## 2.3 Modèle 3 — XGBoost cost-sensitive + Optuna

### Stratégie A — `scale_pos_weight`

Poids relatif des positifs dans les gradients :

\[
\text{scale\_pos\_weight} \approx \frac{n_{neg}}{n_{pos}} \approx 59
\]

Candidats testés : `[10, 30, 59, 77]`.

### Stratégie B — Focal loss personnalisée

\[
FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2,\; \alpha = 0.25
\]

Concentre l'apprentissage sur les **exemples difficiles** (pannes rares mal classées).

### Optimisation Optuna (TPE)

| Paramètre | Plage | Justification |
|-----------|-------|---------------|
| `max_depth` | [3, 8] | Contrôle biais-variance |
| `learning_rate` | [0.01, 0.3] log | Pas d'apprentissage |
| `lambda`, `alpha` | [1e-3, 10] log | Régularisation L2 / L1 |
| `subsample`, `colsample_bytree` | [0.6, 1.0] | Stochasticité / décorrélation des arbres |
| `n_estimators` | [100, 500] | Nombre de boosting rounds |

**Configuration :** `TPESampler(seed=42)`, **50 essais** par stratégie (100 au total), objectif = **maximiser MCC** en CV 5 folds.

### Résultats Stratégie A

| Indicateur | Valeur |
|------------|--------|
| Meilleur `scale_pos_weight` | *[Optuna]* |
| MCC moyen CV | *[study_a.best_value]* |
| Temps | *[min]* |

![Historique Optuna — stratégie A](../reports/figures/optuna_history_strategy_a.png){ width=85% }

![Importance des hyperparamètres — stratégie A](../reports/figures/optuna_importance_strategy_a.png){ width=85% }

### Résultats Stratégie B

| Indicateur | Valeur |
|------------|--------|
| MCC moyen CV | *[study_b.best_value]* |
| Temps | *[min]* |

![Historique Optuna — stratégie B](../reports/figures/optuna_history_strategy_b.png){ width=85% }

![Importance des hyperparamètres — stratégie B](../reports/figures/optuna_importance_strategy_b.png){ width=85% }

### Comparaison A vs B

| Critère | Stratégie A | Stratégie B |
|---------|-------------|-------------|
| MCC CV | *[A]* | *[B]* |
| Δ MCC | *[B − A]* | |
| Simplicité | ✓ native XGBoost | Custom objective |
| Temps | Généralement plus rapide | Souvent plus lent |

**Choix final :** *[retenir la stratégie au meilleur MCC CV ; justifier avec les chiffres Optuna et `xgboost_best.pkl`]*.

### Tableau comparatif des trois modèles

| Modèle | F1-Macro | AUPRC | MCC | Temps (s) |
|--------|----------|-------|-----|-----------|
| Elastic Net | *[ ]* | *[ ]* | *[ ]* | *[ ]* |
| Random Forest | *[ ]* | *[ ]* | *[ ]* | *[ ]* |
| XGBoost (best) | *[ ]* | *[ ]* | *[ ]* | *[ ]* |

*Source recommandée : `reports/tables/model_cv_comparison.csv`.*

![Comparaison des modèles (CV)](../reports/figures/model_cv_comparison.png){ width=90% }

### Conclusion intermédiaire

Le **meilleur modèle global** en validation croisée est en principe **XGBoost** (capacité non linéaire + gestion explicite du déséquilibre). Il sera évalué au chapitre 3 sur le **test set** (16 000 observations) et analysé par SHAP au chapitre 4.

\newpage

# Chapitre 3 — Évaluation et calibration (Étape 3)

## 3.1 Métriques avancées sur le test set

### Pourquoi exclure l'accuracy ?

Avec ≈ 98,3 % de négatifs, un classificateur constant « négatif » maximise l'accuracy tout en générant un coût métier catastrophique (FN).

### Métriques retenues

**F1-Macro** — moyenne non pondérée du F1 par classe :

\[
F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}, \quad F1_{\text{macro}} = \frac{1}{C}\sum_c F1_c
\]

**AUPRC** — aire sous la courbe précision-rappel ; baseline aléatoire ≈ proportion de positifs ({{pos_pct}} %).

**MCC** — corrélation de Matthews (−1 à +1), robuste au déséquilibre :

\[
\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

**Coût IDA 2016 :**

\[
\text{Coût total} = 10 \times FP + 500 \times FN
\]

### Résultats test set

*Insérer le tableau exporté `reports/tables/final_comparison.csv` :*

| Modèle | F1-Macro | AUPRC | MCC | Précision | Rappel | Coût IDA (€) | Seuil |
|--------|----------|-------|-----|-----------|--------|--------------|-------|
| Elastic Net | | | | | | | |
| Random Forest | | | | | | | |
| XGBoost | | | | | | | |

**Meilleur modèle (MCC test) :** *[nom]*.

## 3.2 Courbes de performance

### Precision–Recall

![Courbes Precision-Recall](../reports/figures/pr_curves.png){ width=90% }

*Meilleure AUPRC : [modèle] = [valeur].*

### ROC (référence)

![Courbes ROC](../reports/figures/roc_curves.png){ width=90% }

*Note : en déséquilibre extrême, la ROC peut paraître optimiste ; la PR est prioritaire.*

### MCC et coût vs seuil

Le seuil **0,5** n'est en général **pas optimal**. Le notebook 03 balaie les seuils de 0,10 à 0,90 (pas 0,05).

![MCC vs seuil](../reports/figures/mcc_threshold.png){ width=85% }

- **Seuil optimal MCC :** *[thr_mcc]* — MCC = *[valeur]*.

![Coût métier vs seuil](../reports/figures/cost_threshold.png){ width=85% }

- **Seuil optimal coût :** *[thr_cost]* — Coût minimal = *[€]*.

**Écart MCC / coût :** le seuil minimisant le coût pénalise davantage les FN (×500) ; il peut être **plus bas** que le seuil MCC, augmentant le rappel sur les pannes.

![Barplot métriques finales](../reports/figures/final_metrics_barplot.png){ width=90% }

## 3.3 Calibration des probabilités

### Motivation

Un bon discriminant n'implique pas des **probabilités bien calibrées**. Or l'atelier prend des décisions sur la base de \(P(\text{panne} \mid x)\).

### Reliability diagrams & ECE

**ECE** (Expected Calibration Error) sur 10 bins :

\[
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \mathrm{acc}(b) - \mathrm{conf}(b) \right|
\]

Seuil d'alerte empirique : **ECE > 0,05** → recalibration recommandée.

![Calibration avant recalibration](../reports/figures/calibration_before.png){ width=85% }

![Comparaison calibration avant / après](../reports/figures/calibration_comparison.png){ width=90% }

| Modèle | ECE (avant) | ECE (après) | Méthode |
|--------|-------------|-------------|---------|
| XGBoost | *[ ]* | *[ ]* | Isotonic |
| Random Forest | *[ ]* | *[ ]* | Isotonic |
| Elastic Net | *[ ]* | *[ ]* | Platt (sigmoid) |

**Impact :** comparer F1-Macro, MCC et coût **avant/après** `CalibratedClassifierCV` sur le fold de validation.

**Conclusion :** *[modèle le mieux calibré ; méthode recommandée pour déploiement]*.

\newpage

# Chapitre 4 — Interprétabilité (Étape 4)

## 4.1 SHAP (Shapley Additive Explanations)

### Fondements

SHAP attribue à chaque feature une contribution \(\phi_j\) vérifiant des axiomes de cohérence (Shapley, 1953 ; Lundberg & Lee, 2017) :

\[
f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j
\]

**Explainer :** `shap.TreeExplainer` sur le **XGBoost** retenu.  
**Échantillon :** 5 000 observations du test set (compromis précision / temps).

### Importance globale

![SHAP summary (beeswarm)](../reports/figures/shap_summary.png){ width=95% }

*Top 10 features : voir `reports/tables/shap_feature_importance.csv`.*

![SHAP bar plot](../reports/figures/shap_summary_bar.png){ width=85% }

### Dépendances

![SHAP dependence plots](../reports/figures/shap_dependence.png){ width=95% }

Analyser les **effets non linéaires** et interactions (couleur = feature d'interaction SHAP).

### Explications locales (waterfall)

![Exemples waterfall (TP / FP / FN)](../reports/figures/shap_waterfall_examples.png){ width=95% }

| Cas | Question | Piste d'analyse |
|-----|----------|-----------------|
| **Vrai positif** | Quels capteurs déclenchent l'alerte ? | Features avec \(\phi_j > 0\) élevés |
| **Faux positif** | Pourquoi fausse alerte ? | Valeurs atypiques, bruit |
| **Faux négatif** | Pourquoi panne ratée ? | \(\phi_j\) contradictoires, signal faible |

## 4.2 Stabilité SHAP sur les folds

**Méthodologie :** calcul des valeurs SHAP sur plusieurs splits (ou bootstrap) et mesure de l'**écart-type** par feature.

| Catégorie | Features (exemples) |
|-----------|---------------------|
| Stables (σ faible) | *[liste]* |
| Instables (σ élevé) | *[liste]* |

Les features **stables** constituent des leviers de surveillance capteur plus fiables en production.

## 4.3 Interprétation métier

Les capteurs sont **anonymisés** (`aa_000`, `ab_001`, …) : l'interprétation physique est limitée, mais les patterns restent exploitables :

| Facteur de risque | Signal SHAP | Action opérationnelle suggérée |
|------------------|-------------|------------------------------|
| Feature X élevée | \(\phi_X > 0\) | Renforcer la surveillance du capteur associé |
| Feature Y faible | \(\phi_Y < 0\) sur positifs | Contrôle maintenance préventive |
| Interaction X–W | dépendance non linéaire | Règle composite dans le SCADA |

**Biais potentiels :** corrélations entre capteurs proches physiquement ; dérive temporelle non modélisée (données snapshot).

**Recommandations :**

1. Surveiller en priorité le top-10 SHAP.
2. Déployer le seuil **coût métier** (*[thr_cost]*) plutôt que 0,5 par défaut.
3. Déclencher une inspection APS si \(P(\text{panne}) > \) seuil sur deux fenêtres consécutives.

\newpage

# Conclusion générale

## Synthèse des résultats

| Indicateur | Meilleur modèle / valeur |
|------------|-------------------------|
| Modèle retenu | *[XGBoost / autre]* |
| MCC (test) | *[ ]* |
| AUPRC (test) | *[ ]* |
| ECE (après calibration) | *[ ]* |
| Coût métier minimal | *[ ] €* |
| Seuil opérationnel | *[ ]* |

## Réponse à la problématique

- **Déséquilibre :** gestion multi-niveau (`class_weight`, `scale_pos_weight` / focal loss) avec validation par MCC, pas par accuracy.
- **Calibration :** recalibration isotonique / Platt pour aligner probabilités et risque métier.
- **Interprétabilité :** SHAP identifie les capteurs drivers et les cas d'échec (FN critiques).

## Limites

1. **Généralisation** — données issues d'un sous-ensemble expert Scania ; dérive possible sur d'autres parcs.
2. **Anonymisation** — pas de lien direct capteur ↔ composant physique APS.
3. **Coûts fixes** — 10 € / 500 € simplifient la réalité atelier (main-d'œuvre, immobilisation).
4. **Test set** — labels de test utilisés pour l'évaluation locale ; en compétition, seul le coût agrégé était retourné.

## Perspectives

- **Deep learning** — autoencodeurs pour détection d'anomalies non supervisée en complément.
- **Apprentissage en ligne** — mise à jour du modèle avec flux télématiques.
- **Feature engineering métier** — variables dérivées (gradients pression, écarts à la normale flotte).

## Recommandation de déploiement

| Paramètre | Recommandation |
|-----------|----------------|
| Modèle | *[XGBoost calibré]* |
| Seuil | *[thr_cost]* (coût) ou compromis MCC/coût validé avec le métier |
| Ré-entraînement | Mensuel ou après N nouvelles pannes étiquetées |
| Monitoring | MCC, AUPRC, ECE, coût FP/FN, taux d'alertes |

\newpage

# Bibliographie

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
3. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
4. Zadrozny, B., & Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates.* KDD.
5. McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection.* arXiv.
6. James, G., et al. (2013). *An Introduction to Statistical Learning.* Springer.
7. UCI Machine Learning Repository. *APS Failure and Operational Data at Scania Trucks* (Dataset 421).
8. Costa, C. F., & Nascimento, M. A. (2016). Résultats challenge IDA — coût ≈ 9 920.

\newpage

# Annexes

## Annexe A — Code source

- **Dépôt :** [URL GitHub à compléter]
- **Structure :** `notebooks/01–04`, `src/`, `scripts/generate_all_notebooks.py`
- **Figures :** `reports/figures/` — voir `reports/CORRESPONDANCE_FIGURES.md`

## Annexe B — Installation et reproduction

```bash
git clone <URL_REPO>
cd projet_ai_robuste
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Placer les CSV UCI dans :
# data/raw/aps+failure+at+scania+trucks/

jupyter lab notebooks/01_eda_preprocessing.ipynb
# Puis 02, 03, 04 dans l'ordre

python scripts/extract_report_metrics.py --fill reports/rapport_aps_scania.md
```

## Annexe C — Export PDF

```bash
# Pandoc + LaTeX (MiKTeX / TeX Live)
pandoc reports/rapport_aps_scania_filled.md -o reports/rapport_aps_scania.pdf \
  --from markdown --template eisvogel --toc --number-sections \
  --resource-path=.:reports

# Alternative : VS Code extension "Markdown PDF" ou typst
```

---

*Document généré pour le projet « Classification Robuste — APS Scania ». Remplacer les champs `*[ ]*` et `{{placeholders}}` via `scripts/extract_report_metrics.py` après exécution complète des notebooks.*
