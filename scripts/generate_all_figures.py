#!/usr/bin/env python3
"""Génération automatique des figures de base si les notebooks ne les produisent pas"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import os

PROJECT_ROOT = Path("/content/projet_ai") if 'google.colab' in str(__import__('sys').modules) else Path.cwd()
(PROJECT_ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)

try:
    data_dir = PROJECT_ROOT / "data/raw/aps+failure+at+scania+trucks"
    df = pd.read_csv(data_dir / "aps_failure_training_set.csv", skiprows=20)
    df = df.replace('na', np.nan).apply(pd.to_numeric, errors='coerce')
    y = (df['class'] == 'pos').astype(int)
    X = df.drop(columns=['class'])
except Exception as e:
    print(f"Dataset non charge : {e}")
    exit(1)

# Fig 1: Distribution cible
plt.figure(figsize=(6,4))
plt.bar(['Neg', 'Pos'], [len(y)-y.sum(), y.sum()], color=['#1f77b4','#d62728'])
plt.title('Distribution Cible')
plt.savefig(PROJECT_ROOT/"reports/figures/01_target_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Fig 2: Spearman Top 20
corr = X.corr(method='spearman')
top20 = corr.abs().sum().sort_values(ascending=False).head(20).index
plt.figure(figsize=(10,8))
sns.heatmap(corr.loc[top20, top20], cmap='coolwarm', center=0, square=True)
plt.title('Corrélation Spearman (Top 20)')
plt.savefig(PROJECT_ROOT/"reports/figures/02_correlation_spearman.png", dpi=300, bbox_inches='tight')
plt.close()

print("Figures de base generees")
