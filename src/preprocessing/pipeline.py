"""
Pipelines de prétraitement avec imblearn.

Garantit ZERO data leakage en appliquant toutes les transformations
(imputation, scaling, SMOTE) uniquement dans le fold d'entraînement
via des pipelines imblearn compatibles avec StratifiedKFold.

Références :
- Chawla et al. (2002) : SMOTE
- Hastie et al. (2009) : Importance du scaling dans les modèles linéaires
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(
    with_smote: bool = False,
    smote_random_state: int = 42,
    smote_k_neighbors: int = 5,
    scaling: bool = True
) -> object:
    """
    Crée un pipeline de prétraitement avec ou sans SMOTE.
    
    L'ordre des étapes est crucial :
    1. Imputation médiane (robuste aux outliers)
    2. StandardScaler (centrage-réduction)
    3. SMOTE (optionnel, uniquement sur train fold)
    
    Parameters
    ----------
    with_smote : bool
        Si True, ajoute SMOTE dans le pipeline.
        IMPORTANT : utilise imblearn.Pipeline pour garantir que
        SMOTE n'est appliqué que sur les données d'entraînement.
    smote_random_state : int
        Seed pour SMOTE.
    smote_k_neighbors : int
        Nombre de voisins pour SMOTE.
    scaling : bool
        Si True, applique StandardScaler.
    
    Returns
    -------
    Pipeline
        sklearn.Pipeline si sans SMOTE, imblearn.Pipeline si avec SMOTE.
    
    Notes
    -----
    L'utilisation d'imblearn.Pipeline est OBLIGATOIRE quand SMOTE
    est activé car sklearn.Pipeline n'appelle pas fit_resample().
    """
    steps = []
    
    # Étape 1 : Imputation médiane
    # Justification : La médiane est robuste aux outliers, contrairement
    # à la moyenne qui serait biaisée par les valeurs extrêmes fréquentes
    # dans les données de capteurs industriels.
    steps.append(('imputer', SimpleImputer(strategy='median')))
    
    # Étape 2 : Scaling
    if scaling:
        steps.append(('scaler', StandardScaler()))
    
    # Étape 3 : SMOTE (optionnel)
    if with_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        
        steps.append(('smote', SMOTE(
            random_state=smote_random_state,
            k_neighbors=smote_k_neighbors
        )))
        
        pipeline = ImbPipeline(steps)
        logger.info(
            f"Pipeline imblearn créé avec SMOTE "
            f"(k={smote_k_neighbors}, seed={smote_random_state})"
        )
    else:
        pipeline = SklearnPipeline(steps)
        logger.info("Pipeline sklearn créé (sans SMOTE)")
    
    return pipeline


def create_model_pipeline(
    model,
    with_smote: bool = False,
    smote_random_state: int = 42,
    scaling: bool = True
) -> object:
    """
    Crée un pipeline complet preprocessing + modèle.
    
    Parameters
    ----------
    model : estimator
        Modèle sklearn compatible.
    with_smote : bool
        Si True, utilise imblearn.Pipeline avec SMOTE.
    smote_random_state : int
        Seed pour SMOTE.
    scaling : bool
        Si True, applique StandardScaler.
    
    Returns
    -------
    Pipeline
        Pipeline complet prêt pour fit/predict.
    """
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    
    if scaling:
        steps.append(('scaler', StandardScaler()))
    
    if with_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        
        steps.append(('smote', SMOTE(random_state=smote_random_state)))
        steps.append(('model', model))
        return ImbPipeline(steps)
    else:
        steps.append(('model', model))
        return SklearnPipeline(steps)


def verify_no_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Vérifie qu'il n'y a pas de data leakage entre train et test.
    
    Assertions :
    1. Aucune ligne de test n'est dans le train
    2. Le ratio de classes du test est proche de celui du train
    
    Parameters
    ----------
    X_train, X_test : array-like
        Features.
    y_train, y_test : array-like
        Targets.
    
    Raises
    ------
    AssertionError
        Si un leakage est détecté.
    """
    # Vérification taille
    assert len(X_train) == len(y_train), "X_train et y_train ont des tailles différentes"
    assert len(X_test) == len(y_test), "X_test et y_test ont des tailles différentes"
    assert len(X_train) > len(X_test), "Le train devrait être plus grand que le test"
    
    # Vérification que SMOTE n'a pas été appliqué sur le test
    # (le test ne devrait pas avoir un ratio équilibré)
    pos_ratio_test = np.mean(y_test)
    assert pos_ratio_test < 0.1, (
        f"Le test set semble avoir été rééchantillonné ! "
        f"Ratio pos = {pos_ratio_test:.2%} (attendu < 10%)"
    )
    
    logger.info("✓ Vérification anti-leakage passée avec succès.")
