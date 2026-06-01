#!/usr/bin/env python3
"""Remplit le rapport Markdown avec les métriques réelles"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/content/projet_ai") if 'google.colab' in str(__import__('sys').modules) else Path.cwd()
metrics = {}

final_csv = PROJECT_ROOT / "reports/tables/final_comparison.csv"
if final_csv.exists():
    df = pd.read_csv(final_csv)
    best = df.loc[df['MCC'].idxmax()]
    metrics = {
        'best_model': best['Model'],
        'best_f1': f"{best['F1-Macro']:.4f}",
        'best_auprc': f"{best['AUPRC']:.4f}",
        'best_mcc': f"{best['MCC']:.4f}",
        'optimal_threshold': f"{best.get('Optimal Threshold', 0.5):.2f}"
    }

report_path = PROJECT_ROOT / "rapport_aps_scania.md"
if report_path.exists():
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for k, v in metrics.items():
        content = content.replace(f"{{{{{k}}}}}", v)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Rapport mis à jour ({len(metrics)} valeurs)")
else:
    print("⚠️ rapport_aps_scania.md non trouvé")
