"""
Module de chargement des données APS Failure at Scania Trucks.

Gère le chargement robuste des fichiers CSV avec en-tête de licence,
la conversion des valeurs manquantes, et l'encodage de la cible.

Dataset UCI : https://archive.ics.uci.edu/dataset/421
Challenge IDA 2016

Références :
- Costa & Nascimento (2016) : Score optimal = 9920
- Coût asymétrique : Cost_FP = 10, Cost_FN = 500
"""

import os
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constantes du dataset
N_HEADER_LINES = 20  # Lignes de licence GNU GPL à ignorer
TARGET_COL = 'class'
POS_LABEL = 'pos'
NEG_LABEL = 'neg'
COST_FP = 10   # Coût d'un faux positif (check inutile)
COST_FN = 500  # Coût d'un faux négatif (panne ratée)

# Chemins par défaut (relatifs à la racine du projet)
DEFAULT_DATA_DIR = os.path.join('data', 'raw', 'aps+failure+at+scania+trucks')
TRAIN_FILENAME = 'aps_failure_training_set.csv'
TEST_FILENAME = 'aps_failure_test_set.csv'


def _find_header_line(filepath: str, max_lines: int = 30) -> int:
    """
    Détecte automatiquement la ligne d'en-tête dans le CSV.
    
    Les fichiers APS contiennent une licence GNU GPL en en-tête.
    Cette fonction cherche la première ligne commençant par 'class,'
    qui correspond à l'en-tête des colonnes.
    
    Parameters
    ----------
    filepath : str
        Chemin vers le fichier CSV.
    max_lines : int
        Nombre maximum de lignes à scanner.
    
    Returns
    -------
    int
        Numéro de la ligne d'en-tête (0-indexed).
    
    Raises
    ------
    ValueError
        Si l'en-tête n'est pas trouvé dans les premières lignes.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if line.strip().startswith('class,'):
                logger.info(f"En-tête trouvé à la ligne {i} dans {os.path.basename(filepath)}")
                return i
    
    raise ValueError(
        f"En-tête 'class,...' non trouvé dans les {max_lines} premières lignes "
        f"de {filepath}. Vérifiez le format du fichier."
    )


def load_single_file(
    filepath: str,
    na_values: str = 'na'
) -> pd.DataFrame:
    """
    Charge un fichier CSV APS Scania avec gestion robuste.
    
    - Détecte automatiquement l'en-tête (skip licence)
    - Convertit 'na' → NaN
    - Encode la target : 'pos' → 1, 'neg' → 0
    
    Parameters
    ----------
    filepath : str
        Chemin absolu vers le fichier CSV.
    na_values : str
        Chaîne représentant les valeurs manquantes.
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec target encodée et features numériques.
    
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le format du fichier est invalide.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
    
    logger.info(f"Chargement de {os.path.basename(filepath)}...")
    
    # Détection automatique de la ligne d'en-tête
    header_line = _find_header_line(filepath)
    
    # Chargement avec pandas
    df = pd.read_csv(
        filepath,
        skiprows=header_line,
        na_values=na_values,
        low_memory=False
    )
    
    # Validation de la colonne target
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Colonne '{TARGET_COL}' non trouvée. "
            f"Colonnes disponibles : {list(df.columns[:5])}..."
        )
    
    # Encodage de la target : pos → 1, neg → 0
    label_map = {POS_LABEL: 1, NEG_LABEL: 0}
    unknown_labels = set(df[TARGET_COL].unique()) - set(label_map.keys())
    if unknown_labels:
        raise ValueError(f"Labels inconnus dans la target : {unknown_labels}")
    
    df[TARGET_COL] = df[TARGET_COL].map(label_map)
    
    # Conversion des features en numérique
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(
        f"  Shape: {df.shape} | "
        f"Positifs: {df[TARGET_COL].sum()} ({df[TARGET_COL].mean()*100:.2f}%) | "
        f"NaN total: {df.isna().sum().sum():,}"
    )
    
    return df


def load_aps_data(
    data_dir: Optional[str] = None,
    project_root: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les datasets d'entraînement et de test APS Scania.
    
    Parameters
    ----------
    data_dir : str, optional
        Chemin absolu vers le dossier contenant les fichiers CSV.
        Si None, utilise le chemin par défaut relatif au projet.
    project_root : str, optional
        Racine du projet. Si None, détecté automatiquement.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) avec target encodée.
    
    Examples
    --------
    >>> train_df, test_df = load_aps_data()
    >>> print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    Train: (60000, 171), Test: (16000, 171)
    """
    if data_dir is None:
        if project_root is None:
            # Remonter depuis src/data/ vers la racine du projet
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..')
            )
        data_dir = os.path.join(project_root, DEFAULT_DATA_DIR)
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dossier de données non trouvé : {data_dir}\n"
            f"Assurez-vous que les fichiers APS sont dans {DEFAULT_DATA_DIR}"
        )
    
    train_path = os.path.join(data_dir, TRAIN_FILENAME)
    test_path = os.path.join(data_dir, TEST_FILENAME)
    
    train_df = load_single_file(train_path)
    test_df = load_single_file(test_path)
    
    # Validations
    _validate_data(train_df, test_df)
    
    return train_df, test_df


def _validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Valide la cohérence des données chargées.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Données d'entraînement.
    test_df : pd.DataFrame
        Données de test.
    
    Raises
    ------
    AssertionError
        Si les validations échouent.
    """
    # Vérification des shapes attendues
    assert train_df.shape[1] == test_df.shape[1], (
        f"Nombre de colonnes différent : train={train_df.shape[1]}, test={test_df.shape[1]}"
    )
    assert train_df.shape[1] == 171, (
        f"Nombre de colonnes inattendu : {train_df.shape[1]} (attendu: 171)"
    )
    
    # Vérification de la target
    assert set(train_df[TARGET_COL].unique()) == {0, 1}, (
        f"Valeurs de target inattendues : {train_df[TARGET_COL].unique()}"
    )
    
    # Vérification du déséquilibre
    pos_ratio = train_df[TARGET_COL].mean()
    assert pos_ratio < 0.05, (
        f"Ratio positifs trop élevé : {pos_ratio:.2%} (attendu < 5%)"
    )
    
    # Log du résumé
    n_pos_train = train_df[TARGET_COL].sum()
    n_neg_train = (train_df[TARGET_COL] == 0).sum()
    ratio = n_neg_train / n_pos_train if n_pos_train > 0 else float('inf')
    
    logger.info("=" * 60)
    logger.info("RÉSUMÉ DU CHARGEMENT")
    logger.info(f"  Train : {train_df.shape[0]:,} lignes ({n_pos_train:,} pos / {n_neg_train:,} neg)")
    logger.info(f"  Test  : {test_df.shape[0]:,} lignes")
    logger.info(f"  Ratio déséquilibre : 1:{ratio:.0f}")
    logger.info(f"  Features : {train_df.shape[1] - 1}")
    logger.info(f"  NaN train : {train_df.isna().sum().sum():,}")
    logger.info(f"  NaN test  : {test_df.isna().sum().sum():,}")
    logger.info("=" * 60)


def get_X_y(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare features et target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec colonne 'class'.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) — features et target.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def compute_total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> int:
    """
    Calcule le coût total selon la métrique du challenge IDA 2016.
    
    Total_cost = Cost_FP * n_FP + Cost_FN * n_FN
    avec Cost_FP = 10 et Cost_FN = 500.
    
    Parameters
    ----------
    y_true : array-like
        Labels réels.
    y_pred : array-like
        Labels prédits.
    
    Returns
    -------
    int
        Coût total.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    fp = np.sum((y_pred == 1) & (y_true == 0))  # Faux positifs
    fn = np.sum((y_pred == 0) & (y_true == 1))  # Faux négatifs
    
    total_cost = COST_FP * fp + COST_FN * fn
    
    logger.info(f"Coût total: {total_cost:,} (FP={fp}, FN={fn})")
    return total_cost
