"""
CasADi Integration Module

This module provides the bridge between CasADi NLP formulations and OPOCH's
proof-carrying optimization kernel.

Two Contracts:
- Contract L (Local-KKT): Certified KKT residuals for local optimality
- Contract G (Global Proof): UB-LB gap closure for global optimality

Every case ends in UNIQUE / UNSAT / Ω with replayable receipts.
"""

from .nlp_contract import CasADiNLP, NLPBounds
from .adapter import CasADiAdapter
from .kkt_certificate import KKTCertificate, KKTCertifier, KKTStatus
from .solver_wrapper import DeterministicIPOPT, DeterministicBonmin
from .global_certificate import GlobalCertificate, GlobalCertifier

__all__ = [
    # Core NLP representation
    'CasADiNLP',
    'NLPBounds',
    # Adapter
    'CasADiAdapter',
    # Contract L (KKT)
    'KKTCertificate',
    'KKTCertifier',
    'KKTStatus',
    # Solvers
    'DeterministicIPOPT',
    'DeterministicBonmin',
    # Contract G (Global)
    'GlobalCertificate',
    'GlobalCertifier',
]
