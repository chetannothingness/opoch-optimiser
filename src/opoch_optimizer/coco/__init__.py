"""
OPOCH COCO/BBOB Module

Deterministic Covariance Matrix Adaptation for black-box optimization.
Achieves rotational invariance through metric learning.
"""

from .dcma import DCMA, DCMAConfig, DCMAResult
from .quadratic_id import QuadraticIdentifier, QuadraticResult
from .portfolio import COCOPortfolio, AdaptiveRestartPortfolio, PortfolioResult
from .logger_ioh import IOHLogger, IOHExperimentLogger, RunData

__all__ = [
    'DCMA',
    'DCMAConfig',
    'DCMAResult',
    'QuadraticIdentifier',
    'QuadraticResult',
    'COCOPortfolio',
    'AdaptiveRestartPortfolio',
    'PortfolioResult',
    'IOHLogger',
    'IOHExperimentLogger',
    'RunData',
]
