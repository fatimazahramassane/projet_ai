# Correspondance figures — énoncé ↔ projet

Les notebooks enregistrent les figures sous `reports/figures/` avec des noms numérotés.

| Figure demandée (énoncé) | Fichier généré par le projet |
|-------------------------|------------------------------|
| `distribution_cible.png` | `01_class_distribution.png` |
| `missing_values.png` (optionnel) | `02_missing_values.png` |
| `correlation_spearman.png` | `04_correlation_heatmap.png` |
| `correlation_target.png` | `05_target_correlation.png` |
| `top_10_vif_scores` (tableau) | `reports/tables/top_10_vif_scores.csv` (via `extract_report_metrics.py`) |
| `violin_top_features.png` | `07_violin_plots.png` |
| `umap_projection.png` | `08_umap_2d.png` |
| `imbalance_strategy_comparison.png` | `09_imbalance_comparison.png` |
| `rf_proximity_outliers.png` | `rf_mds_proximity_outliers.png` |
| `optuna_strategyA_history.png` | `optuna_history_strategy_a.png` |
| `optuna_strategyA_params.png` | `optuna_importance_strategy_a.png`, `optuna_slice_strategy_a.png` |
| `optuna_strategyB_history.png` | `optuna_history_strategy_b.png` |
| `optuna_strategyB_params.png` | `optuna_importance_strategy_b.png`, `optuna_slice_strategy_b.png` |
| `model_comparison_barplot.png` | `model_cv_comparison.png` (CV) + `final_metrics_barplot.png` (test) |
| `pr_curves.png` | `pr_curves.png` |
| `roc_curves.png` | `roc_curves.png` |
| `mcc_threshold.png` | `mcc_threshold.png` |
| `cost_threshold.png` | `cost_threshold.png` |
| `calibration_before_after.png` | `calibration_before.png` + `calibration_comparison.png` |
| `shap_summary_beeswarm.png` | `shap_summary.png` |
| `shap_summary_bar.png` | `shap_summary_bar.png` |
| `shap_dependence_*.png` | `shap_dependence.png` (multi-panels) |
| `shap_waterfall_*.png` | `shap_waterfall_examples.png` |

**Tableaux :**

| Fichier énoncé | Fichier projet |
|----------------|----------------|
| `final_comparison.csv` | `reports/tables/final_comparison.csv` |
| `model_cv_comparison.csv` | `reports/tables/model_cv_comparison.csv` |
| `shap_feature_importance.csv` | `reports/tables/shap_feature_importance.csv` |
