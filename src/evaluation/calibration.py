"""
Module de calibration des probabilités.

La calibration garantit que les probabilités de sortie du modèle
correspondent à la réalité statistique. Un modèle bien calibré
qui prédit P=0.8 devrait avoir raison ~80% du temps.

Méthodes :
- Platt Scaling (régression logistique sur les logits) : adapté aux
  modèles à sortie sigmoïde (SVM, boosting).
- Isotonic Regression : non-paramétrique, plus flexible mais
  nécessite plus de données.

Références :
- Platt (1999) : "Probabilistic Outputs for SVMs"
- Niculescu-Mizil & Caruana (2005) : comparaison des méthodes
"""

import logging
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

logger = logging.getLogger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calcule l'Expected Calibration Error (ECE).
    
    ECE = Σ (|B_k|/N) * |acc(B_k) - conf(B_k)|
    
    où B_k sont les bins, acc est la précision réelle,
    et conf est la confiance moyenne du modèle.
    
    Parameters
    ----------
    y_true : array-like
        Labels réels.
    y_proba : array-like
        Probabilités prédites.
    n_bins : int
        Nombre de bins.
    
    Returns
    -------
    float
        ECE entre 0 et 1 (plus bas = mieux calibré).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        
        bin_acc = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        
        ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    model_name: str = "Modèle",
    ax: Optional[plt.Axes] = None,
    save_name: Optional[str] = None
) -> Tuple[plt.Figure, float]:
    """
    Trace un diagramme de fiabilité (Reliability Diagram).
    
    Parameters
    ----------
    y_true : array-like
        Labels réels.
    y_proba : array-like
        Probabilités prédites.
    n_bins : int
        Nombre de bins.
    model_name : str
        Nom du modèle.
    ax : plt.Axes, optional
        Axes matplotlib.
    save_name : str, optional
        Nom pour sauvegarde.
    
    Returns
    -------
    tuple
        (fig, ece)
    """
    from src.evaluation.plots import setup_plot_style, save_figure, COLORS
    setup_plot_style()
    
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    ece = compute_ece(y_true, y_proba, n_bins)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Diagonale parfaite
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Calibration parfaite')
    
    # Courbe de calibration
    ax.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8,
            color=COLORS['primary'],
            label=f'{model_name} (ECE={ece:.4f})')
    
    # Histogramme des probabilités en fond
    ax2 = ax.twinx()
    ax2.hist(y_proba, bins=n_bins, range=(0, 1), alpha=0.15,
             color=COLORS['secondary'], edgecolor='none')
    ax2.set_ylabel('Nombre de prédictions', alpha=0.5)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 3)
    
    ax.set_xlabel('Probabilité prédite (moyenne par bin)')
    ax.set_ylabel('Proportion de positifs (réelle)')
    ax.set_title(f'Diagramme de fiabilité — {model_name}', fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig, ece


def calibrate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'isotonic',
    cv: int = 5
) -> CalibratedClassifierCV:
    """
    Calibre un modèle avec Platt Scaling ou Isotonic Regression.
    
    Parameters
    ----------
    model : estimator
        Modèle entraîné.
    X_train : array-like
        Features d'entraînement.
    y_train : array-like
        Target d'entraînement.
    method : str
        'sigmoid' (Platt) ou 'isotonic'.
    cv : int
        Nombre de folds pour la calibration.
    
    Returns
    -------
    CalibratedClassifierCV
        Modèle calibré.
    """
    logger.info(f"Calibration du modèle avec méthode '{method}' (cv={cv})...")
    
    calibrated = CalibratedClassifierCV(
        model, method=method, cv=cv
    )
    calibrated.fit(X_train, y_train)
    
    logger.info("Calibration terminée.")
    return calibrated
