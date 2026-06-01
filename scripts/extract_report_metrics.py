"""
Extrait les métriques des notebooks (EDA, tables CSV, modèles) pour le rapport.

Usage (depuis la racine du projet) :
    python scripts/extract_report_metrics.py
    python scripts/extract_report_metrics.py --fill reports/rapport_aps_scania.md

Produit :
    reports/metrics_summary.json
    reports/tables/top_10_vif_scores.csv  (si eda_results.joblib présent)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
OUT_JSON = ROOT / "reports" / "metrics_summary.json"


def _load_eda():
    path = PROCESSED / "eda_results.joblib"
    if not path.exists():
        return {}
    import joblib

    eda = joblib.load(path)
    out = {}
    comp = eda.get("comparison_scores", {})
    for strat, key in [("class_weight", "A"), ("smote", "B")]:
        s = comp.get(strat, {})
        if not s:
            continue
        import numpy as np

        f1 = np.array(s.get("f1_macro", []))
        mcc = np.array(s.get("mcc", []))
        if len(f1):
            out[f"imbalance_{key}_f1_mean"] = float(f1.mean())
            out[f"imbalance_{key}_f1_std"] = float(f1.std())
        if len(mcc):
            out[f"imbalance_{key}_mcc_mean"] = float(mcc.mean())
            out[f"imbalance_{key}_mcc_std"] = float(mcc.std())

    vif_df = eda.get("vif_df")
    if vif_df is not None:
        high = vif_df[vif_df["VIF"] > 5]
        out["vif_count_gt5"] = int(len(high))
        out["vif_top10_features"] = (
            vif_df.sort_values("VIF", ascending=False)
            .head(10)["Feature"]
            .tolist()
        )
        top_path = TABLES / "top_10_vif_scores.csv"
        TABLES.mkdir(parents=True, exist_ok=True)
        vif_df.sort_values("VIF", ascending=False).head(10).to_csv(top_path, index=False)

    top_f = eda.get("top_features")
    if top_f is not None:
        out["top5_discriminative_features"] = list(top_f)[:5]

    return out


def _load_tables():
    out = {}
    for name, key in [
        ("final_comparison.csv", "test_metrics"),
        ("model_cv_comparison.csv", "cv_metrics"),
        ("shap_feature_importance.csv", "shap_top"),
    ]:
        path = TABLES / name
        if not path.exists():
            continue
        import pandas as pd

        df = pd.read_csv(path)
        out[key] = df.to_dict(orient="records")
    return out


def _load_models_quick():
    """Évaluation test rapide si les .pkl existent."""
    if not MODELS.exists():
        return {}
    pkl = list(MODELS.glob("*.pkl"))
    if not pkl:
        return {}
    try:
        from src.data.loader import load_aps_data, get_X_y
        import joblib
        from src.evaluation.metrics import compute_all_metrics, find_optimal_threshold

        train_df, test_df = load_aps_data(project_root=str(ROOT))
        X_train, y_train = get_X_y(train_df)
        X_test, y_test = get_X_y(test_df)
    except Exception as e:
        return {"model_eval_error": str(e)}

    out = {}
    mapping = {
        "logreg_elasticnet.pkl": "Elastic Net",
        "random_forest.pkl": "Random Forest",
        "xgboost_best.pkl": "XGBoost",
    }
    for fname, label in mapping.items():
        path = MODELS / fname
        if not path.exists():
            continue
        obj = joblib.load(path)
        if fname == "xgboost_best.pkl" and isinstance(obj, dict):
            model = obj.get("model") or obj.get("estimator")
            proba_fn = obj.get("predict_proba")
            if proba_fn is None and model is not None:
                proba = model.predict_proba(X_test)[:, 1]
            elif callable(proba_fn):
                proba = proba_fn(X_test)
            else:
                continue
        elif hasattr(obj, "predict_proba"):
            proba = obj.predict_proba(X_test)[:, 1]
        else:
            continue
        thr, _, _, _ = find_optimal_threshold(
            y_train.values, obj.predict_proba(X_train)[:, 1]
            if hasattr(obj, "predict_proba")
            else proba,
            metric="mcc",
        )
        pred = (proba >= thr).astype(int)
        import numpy as np

        m = compute_all_metrics(y_test.values, pred, proba)
        out[label] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}
        out[label]["threshold"] = float(thr)
    return {"test_eval": out}


def build_summary() -> dict:
    summary = {
        "project_root": str(ROOT),
        "figures_found": sorted(p.name for p in FIGURES.glob("*.png")),
    }
    summary.update(_load_eda())
    summary.update(_load_tables())
    summary.update(_load_models_quick())

    try:
        from src.data.loader import load_aps_data, get_X_y

        train_df, test_df = load_aps_data(project_root=str(ROOT))
        X, y = get_X_y(train_df)
        summary["train_shape"] = list(train_df.shape)
        summary["test_shape"] = list(test_df.shape)
        summary["n_pos"] = int(y.sum())
        summary["n_neg"] = int((y == 0).sum())
        summary["pos_pct"] = round(float(y.mean()) * 100, 2)
        summary["scale_pos_weight"] = round(float((y == 0).sum() / max(y.sum(), 1)), 1)
        miss = X.isna().mean()
        summary["mean_missing_pct"] = round(float(miss.mean()) * 100, 2)
        summary["features_with_missing_gt50pct"] = int((miss > 0.5).sum())
    except Exception as e:
        summary["data_load_error"] = str(e)

    return summary


def fill_template(md_path: Path, summary: dict) -> str:
    flat = {}

    def _flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            flat[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            flat[prefix] = obj

    _flatten("", summary)

    text = md_path.read_text(encoding="utf-8")

    def repl(match):
        key = match.group(1).strip()
        if key in flat:
            val = flat[key]
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)
        return match.group(0)

    return re.sub(r"\{\{([^}]+)\}\}", repl, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fill",
        type=Path,
        default=None,
        help="Génère une copie du rapport avec les {{placeholders}} remplis",
    )
    args = parser.parse_args()

    summary = build_summary()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"Écrit : {OUT_JSON}")
    print(f"Figures PNG trouvées : {len(summary.get('figures_found', []))}")

    if args.fill:
        out_path = args.fill.with_name(args.fill.stem + "_filled.md")
        out_path.write_text(fill_template(args.fill, summary), encoding="utf-8")
        print(f"Rapport rempli : {out_path}")


if __name__ == "__main__":
    main()
