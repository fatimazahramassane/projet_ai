"""
Métriques d'évaluation pour classification déséquilibrée.

Justification du choix des métriques :
- F1-Macro : Moyenne non-pondérée du F1 par classe, traite chaque classe
  également indépendamment de sa fréquence.
- AUPRC : Aire sous la courbe Precision-Recall, plus informative que
  AUC-ROC quand les classes sont très déséquilibrées (Davis & Goadrich, 2006).
- MCC : Coefficient de Matthews, seule métrique qui prend en compte les
  4 cellules de la matrice de confusion (Chicco & Jurman, 2020).

L'Accuracy est EXCLUE car elle est trompeuse avec des classes déséquilibrées
(un modèle prédisant toujours 'neg' aurait ~98.3% d'accuracy).
"""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcule toutes les métriques de classification.
    
    Parameters
    ----------
    y_true : array-like
        Labels réels.
    y_pred : array-like
        Labels prédits.
    y_proba : array-like, optional
        Probabilités prédites pour la classe positive.
    
    Returns
    -------
    dict
        Dictionnaire de métriques.
    """
    metrics = {
        'F1-Macro': f1_score(y_true, y_pred, average='macro'),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Binary': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_proba is not None:
        metrics['AUPRC'] = average_precision_score(y_true, y_proba)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['TP'] = int(tp)
    metrics['FP'] = int(fp)
    metrics['FN'] = int(fn)
    metrics['TN'] = int(tn)
    
    # Coût total (métrique IDA 2016)
    metrics['Cost'] = 10 * fp + 500 * fn
    
    return metrics


def print_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Modèle"
) -> Dict[str, float]:
    """
    Affiche un rapport complet des métriques.
    
    Parameters
    ----------
    y_true : array-like
        Labels réels.
    y_pred : array-like
        Labels prédits.
    y_proba : array-like, optional
        Probabilités prédites.
    model_name : str
        Nom du modèle pour l'affichage.
    
    Returns
    -------
    dict
        Dictionnaire de métriques.
    """
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION : {model_name}")
    print(f"{'='*60}")
    print(f"  F1-Macro  : {metrics['F1-Macro']:.4f}")
    print(f"  MCC       : {metrics['MCC']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    if 'AUPRC' in metrics:
        print(f"  AUPRC     : {metrics['AUPRC']:.4f}")
    print(f"  Coût IDA  : {metrics['Cost']:,}")
    print(f"  TP={metrics['TP']} | FP={metrics['FP']} | FN={metrics['FN']} | TN={metrics['TN']}")
    print(f"{'='*60}\n")
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'mcc',
    thresholds: Optional[np.ndarray] = None
) -> tuple:
    """
    Trouve le seuil optimal pour une métrique donnée.
    
    Parameters
    ----------
    y_true : array-like
        Labels réels.
    y_proba : array-like
        Probabilités prédites.
    metric : str
        Métrique à optimiser ('mcc', 'f1', 'cost').
    thresholds : array-like, optional
        Seuils à évaluer. Par défaut: 0.01 à 0.99.
    
    Returns
    -------
    tuple
        (optimal_threshold, optimal_score, all_thresholds, all_scores)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    scores = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        
        if metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred_t)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred_t, average='macro')
        elif metric == 'cost':
            cm = confusion_matrix(y_true, y_pred_t)
            tn, fp, fn, tp = cm.ravel()
            score = -(10 * fp + 500 * fn)  # Négatif car on maximise
        else:
            raise ValueError(f"Métrique inconnue : {metric}")
        
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    
    return thresholds[best_idx], scores[best_idx], thresholds, scores
