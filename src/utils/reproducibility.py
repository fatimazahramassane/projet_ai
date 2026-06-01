"""
Module de reproductibilité et logging.

Garantit la reproductibilité totale des expériences en fixant
tous les seeds et en loggant les versions des librairies.
"""

import os
import random
import logging
import platform
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

GLOBAL_SEED = 42


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """
    Fixe tous les seeds pour garantir la reproductibilité.
    
    Parameters
    ----------
    seed : int
        Valeur du seed (défaut: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import xgboost as xgb
        # XGBoost uses seed parameter in model params
        logger.info(f"XGBoost disponible, seed sera fixé dans les paramètres du modèle.")
    except ImportError:
        logger.warning("XGBoost non installé.")
    
    logger.info(f"Tous les seeds fixés à {seed}.")


def log_environment_info() -> dict:
    """
    Enregistre et retourne les informations de l'environnement.
    
    Returns
    -------
    dict
        Dictionnaire avec les versions de toutes les librairies.
    """
    import sklearn
    import pandas as pd
    import matplotlib
    import scipy
    
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': sklearn.__version__,
        'scipy': scipy.__version__,
        'matplotlib': matplotlib.__version__,
    }
    
    # Optional libraries
    optional_libs = [
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('imbalanced-learn', 'imblearn'),
        ('optuna', 'optuna'),
        ('shap', 'shap'),
        ('umap-learn', 'umap'),
        ('seaborn', 'seaborn'),
    ]
    
    for name, module_name in optional_libs:
        try:
            mod = __import__(module_name)
            env_info[name] = mod.__version__
        except ImportError:
            env_info[name] = 'Non installé'
    
    logger.info("=" * 60)
    logger.info("ENVIRONNEMENT D'EXÉCUTION")
    logger.info("=" * 60)
    for key, value in env_info.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    return env_info


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure le logging pour le projet.
    
    Parameters
    ----------
    level : int
        Niveau de logging.
    log_file : str, optional
        Chemin vers un fichier de log.
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger.info("Logging configuré.")


def get_cv_splitter(n_splits: int = 5, seed: int = GLOBAL_SEED):
    """
    Retourne un StratifiedKFold configuré pour la reproductibilité.
    
    Parameters
    ----------
    n_splits : int
        Nombre de folds.
    seed : int
        Seed pour le shuffle.
    
    Returns
    -------
    StratifiedKFold
    """
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
