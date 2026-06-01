"""
Fonctions de visualisation réutilisables pour le projet.

Toutes les figures sont sauvegardées en PNG 300 DPI dans reports/figures/.
Style cohérent avec palette de couleurs définie.
"""

import os
import logging
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    confusion_matrix,
    matthews_corrcoef
)

logger = logging.getLogger(__name__)

# Configuration globale du style
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
}

# Palette de couleurs cohérente
COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FFC107',
    'info': '#00BCD4',
    'neg_class': '#2196F3',
    'pos_class': '#F44336',
    'outlier': '#FF5722',
}

DEFAULT_SAVE_DIR = os.path.join('reports', 'figures')


def setup_plot_style():
    """Configure le style global des graphiques."""
    plt.rcParams.update(PLOT_STYLE)
    sns.set_style('whitegrid')
    sns.set_palette('husl')


def save_figure(
    fig: plt.Figure,
    filename: str,
    save_dir: str = DEFAULT_SAVE_DIR,
    dpi: int = 300
) -> str:
    """
    Sauvegarde une figure en PNG haute résolution.
    
    Parameters
    ----------
    fig : matplotlib.Figure
        Figure à sauvegarder.
    filename : str
        Nom du fichier (sans extension).
    save_dir : str
        Dossier de sauvegarde.
    dpi : int
        Résolution.
    
    Returns
    -------
    str
        Chemin complet du fichier sauvegardé.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{filename}.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"Figure sauvegardée : {filepath}")
    return filepath


def plot_class_distribution(
    y: pd.Series,
    title: str = "Distribution des classes",
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Barplot de la distribution des classes.
    
    Parameters
    ----------
    y : pd.Series
        Target (0/1).
    title : str
        Titre du graphique.
    save_name : str, optional
        Nom du fichier pour sauvegarde.
    
    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Barplot
    counts = y.value_counts().sort_index()
    labels = ['Négatif (pas APS)', 'Positif (panne APS)']
    colors = [COLORS['neg_class'], COLORS['pos_class']]
    
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=2)
    for bar, count in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.02,
            f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=13
        )
    axes[0].set_ylabel('Nombre d\'observations')
    axes[0].set_title(title)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct='%1.2f%%', startangle=90,
        explode=(0, 0.1), shadow=True,
        textprops={'fontsize': 12}
    )
    axes[1].set_title('Proportion des classes')
    
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig


def plot_missing_values(
    df: pd.DataFrame,
    top_n: int = 30,
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Barplot horizontal des features avec le plus de valeurs manquantes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.
    top_n : int
        Nombre de features à afficher.
    save_name : str, optional
        Nom du fichier pour sauvegarde.
    
    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()
    
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(missing_pct) * 0.35)))
    
    colors = plt.cm.RdYlGn_r(missing_pct.values / missing_pct.values.max())
    
    bars = ax.barh(range(len(missing_pct)), missing_pct.values, color=colors)
    ax.set_yticks(range(len(missing_pct)))
    ax.set_yticklabels(missing_pct.index, fontsize=9)
    ax.set_xlabel('% de valeurs manquantes')
    ax.set_title(f'Top {top_n} features avec valeurs manquantes')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Annotations
    for i, (val, name) in enumerate(zip(missing_pct.values, missing_pct.index)):
        ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Matrice de corrélation (Spearman)",
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Heatmap de la matrice de corrélation.
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Matrice de corrélation.
    title : str
        Titre.
    save_name : str, optional
        Nom du fichier.
    
    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()
    
    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(10, n*0.5), max(8, n*0.4)))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Corrélation'},
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig


def plot_violin_by_class(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'class',
    n_cols: int = 3,
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Violin plots des features par classe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec features et target.
    features : list[str]
        Liste des features à visualiser.
    target_col : str
        Nom de la colonne target.
    n_cols : int
        Nombre de colonnes dans la grille.
    save_name : str, optional
        Nom du fichier.
    
    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()
    
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes)
    
    palette = {0: COLORS['neg_class'], 1: COLORS['pos_class']}
    
    for idx, feature in enumerate(features):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        sns.violinplot(
            data=df, x=target_col, y=feature,
            palette=palette, ax=ax, inner='box',
            cut=0, density_norm='width'
        )
        ax.set_title(feature, fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(['Négatif', 'Positif'])
    
    # Masquer les axes vides
    for idx in range(n_features, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)
    
    fig.suptitle('Distribution des features par classe', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Modèle",
    ax: Optional[plt.Axes] = None,
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Courbe Precision-Recall avec AUPRC annoté.
    """
    setup_plot_style()
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AUPRC={auprc:.4f})')
    ax.fill_between(recall, precision, alpha=0.1)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Courbe Precision-Recall')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig


def plot_mcc_vs_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    save_name: Optional[str] = None
) -> Tuple[plt.Figure, float, float]:
    """
    MCC en fonction du seuil de décision.
    
    Returns
    -------
    tuple
        (fig, optimal_threshold, optimal_mcc)
    """
    setup_plot_style()
    
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    mcc_scores = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        mcc_scores.append(matthews_corrcoef(y_true, y_pred_t))
    
    mcc_scores = np.array(mcc_scores)
    best_idx = np.argmax(mcc_scores)
    best_threshold = thresholds[best_idx]
    best_mcc = mcc_scores[best_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, mcc_scores, linewidth=2, color=COLORS['primary'])
    ax.axvline(best_threshold, color=COLORS['danger'], linestyle='--', linewidth=1.5,
               label=f'Seuil optimal = {best_threshold:.2f}')
    ax.scatter([best_threshold], [best_mcc], color=COLORS['danger'], s=100, zorder=5)
    ax.annotate(
        f'MCC max = {best_mcc:.4f}\nSeuil = {best_threshold:.2f}',
        xy=(best_threshold, best_mcc),
        xytext=(best_threshold + 0.1, best_mcc - 0.1),
        fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black')
    )
    ax.set_xlabel('Seuil de décision')
    ax.set_ylabel('MCC (Matthews Correlation Coefficient)')
    ax.set_title('MCC vs Seuil de décision')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig, best_threshold, best_mcc


def plot_dimensionality_reduction(
    embedding: np.ndarray,
    labels: np.ndarray,
    method_name: str = "UMAP",
    title: Optional[str] = None,
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot 2D de la réduction de dimension.
    
    Parameters
    ----------
    embedding : ndarray, shape (n_samples, 2)
        Coordonnées 2D.
    labels : ndarray
        Classes (0/1).
    method_name : str
        Nom de la méthode (UMAP, t-SNE).
    title : str, optional
        Titre personnalisé.
    save_name : str, optional
        Nom du fichier.
    
    Returns
    -------
    matplotlib.Figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Négatifs d'abord (fond), positifs ensuite (devant)
    mask_neg = labels == 0
    mask_pos = labels == 1
    
    ax.scatter(
        embedding[mask_neg, 0], embedding[mask_neg, 1],
        c=COLORS['neg_class'], label='Négatif', alpha=0.3, s=10
    )
    ax.scatter(
        embedding[mask_pos, 0], embedding[mask_pos, 1],
        c=COLORS['pos_class'], label='Positif (APS)', alpha=0.8, s=30,
        edgecolors='black', linewidths=0.5
    )
    
    if title is None:
        title = f'Projection {method_name} 2D — colorée par classe'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method_name} dimension 1')
    ax.set_ylabel(f'{method_name} dimension 2')
    ax.legend(fontsize=12, markerscale=2)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig
