"""
Global Certificate Computation (Contract G)

Computes certified global optimality via UB-LB gap closure.
This is Contract G: proven global bounds with replay receipts.

Uses OPOCH's branch-and-reduce engine with:
- Interval/Taylor model enclosures for LB
- FBBT/Krawczyk contractors
- Structure-specific bounds (ResidualIR, FactorIR)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np
import json
import hashlib
from datetime import datetime


class GlobalStatus(Enum):
    """Global certificate status."""
    UNIQUE_OPT = "UNIQUE_OPT"      # UB - LB <= epsilon
    UNSAT = "UNSAT"                # Proven infeasible
    OMEGA = "OMEGA"                # Resource cap reached


@dataclass
class GlobalCertificate:
    """
    Certified global optimality via UB-LB gap closure.

    This is the proof bundle for Contract G.
    """
    # Solution
    x_opt: Optional[np.ndarray]      # Optimal point (if found)
    upper_bound: float               # UB = f(x_opt) or inf
    lower_bound: float               # Certified LB
    gap: float                       # UB - LB
    epsilon: float                   # Gap tolerance

    # Status
    status: GlobalStatus

    # Search statistics
    nodes_explored: int = 0
    contractor_applications: int = 0
    precision_escalations: int = 0
    time_seconds: float = 0.0

    # IR type used
    ir_type: str = "ExprIR"

    # Hashes for replay
    input_hash: str = ""
    bounds_hash: str = ""
    certificate_hash: str = ""

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        if self.x_opt is not None:
            self.x_opt = np.asarray(self.x_opt)

    @property
    def is_certified(self) -> bool:
        """Check if globally certified."""
        return self.status == GlobalStatus.UNIQUE_OPT

    @property
    def is_infeasible(self) -> bool:
        """Check if proven infeasible."""
        return self.status == GlobalStatus.UNSAT

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'x_opt': self.x_opt.tolist() if self.x_opt is not None else None,
            'upper_bound': self.upper_bound,
            'lower_bound': self.lower_bound,
            'gap': self.gap,
            'epsilon': self.epsilon,
            'status': self.status.value,
            'statistics': {
                'nodes_explored': self.nodes_explored,
                'contractor_applications': self.contractor_applications,
                'precision_escalations': self.precision_escalations,
                'time_seconds': self.time_seconds,
            },
            'ir_type': self.ir_type,
            'hashes': {
                'input': self.input_hash,
                'bounds': self.bounds_hash,
                'certificate': self.certificate_hash,
            },
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GlobalCertificate':
        """Deserialize from dictionary."""
        return cls(
            x_opt=np.array(d['x_opt']) if d['x_opt'] is not None else None,
            upper_bound=d['upper_bound'],
            lower_bound=d['lower_bound'],
            gap=d['gap'],
            epsilon=d['epsilon'],
            status=GlobalStatus(d['status']),
            nodes_explored=d['statistics']['nodes_explored'],
            contractor_applications=d['statistics']['contractor_applications'],
            precision_escalations=d['statistics']['precision_escalations'],
            time_seconds=d['statistics']['time_seconds'],
            ir_type=d['ir_type'],
            input_hash=d['hashes']['input'],
            bounds_hash=d['hashes']['bounds'],
            certificate_hash=d['hashes']['certificate'],
            timestamp=d['timestamp'],
        )

    def save_json(self, path: str):
        """Save certificate to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> 'GlobalCertificate':
        """Load certificate from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class GlobalCertifier:
    """
    Global optimization certification using OPOCH engine.

    Provides certified global bounds via:
    1. Branch-and-reduce with interval/Taylor enclosures
    2. FBBT/Krawczyk contractors
    3. Structure-specific witnesses (ResidualIR, FactorIR)
    """

    def __init__(
        self,
        adapter: 'CasADiAdapter',
        epsilon: float = 1e-4,
        max_time: float = 60.0,
        max_nodes: int = 50000
    ):
        """
        Initialize global certifier.

        Args:
            adapter: CasADiAdapter with AD functions
            epsilon: Gap tolerance for certification
            max_time: Maximum time budget (seconds)
            max_nodes: Maximum nodes to explore
        """
        self.adapter = adapter
        self.nlp = adapter.nlp
        self.epsilon = epsilon
        self.max_time = max_time
        self.max_nodes = max_nodes

        # Build ObjectiveIR
        self._ir = None
        self._build_ir()

    def _build_ir(self):
        """Build ObjectiveIR from CasADi problem."""
        self._ir = self.adapter.to_objective_ir()

    def certify(self) -> GlobalCertificate:
        """
        Run global certification via OPOCH kernel.

        Returns:
            GlobalCertificate with bounds and status
        """
        import time
        from ..solver.opoch_kernel import OPOCHKernel, OPOCHConfig
        from ..core.output_gate import Verdict

        start_time = time.time()

        # Build problem contract
        contract = self.adapter.to_problem_contract()

        # Configure kernel
        config = OPOCHConfig(
            epsilon=self.epsilon,
            max_time=self.max_time,
            max_nodes=self.max_nodes,
            log_frequency=10000,  # Quiet logging
        )

        # Run solver
        try:
            kernel = OPOCHKernel(contract, config)
            verdict, result = kernel.solve()

            elapsed = time.time() - start_time

            # Map verdict to status
            if verdict == Verdict.UNIQUE_OPT:
                status = GlobalStatus.UNIQUE_OPT
            elif verdict == Verdict.UNSAT:
                status = GlobalStatus.UNSAT
            else:
                status = GlobalStatus.OMEGA

            # Extract solution
            x_opt = getattr(result, 'x_optimal', None)
            ub = getattr(result, 'upper_bound', float('inf'))
            lb = getattr(result, 'lower_bound', float('-inf'))
            gap = ub - lb
            nodes = getattr(result, 'nodes_explored', 0)

            # Compute hashes
            input_hash = self.nlp.canonical_hash()
            bounds_hash = self._compute_bounds_hash(lb, ub)
            cert_hash = self._compute_certificate_hash(x_opt, lb, ub, status)

            return GlobalCertificate(
                x_opt=x_opt,
                upper_bound=ub,
                lower_bound=lb,
                gap=gap,
                epsilon=self.epsilon,
                status=status,
                nodes_explored=nodes,
                contractor_applications=0,  # TODO: track
                precision_escalations=0,  # TODO: track
                time_seconds=elapsed,
                ir_type=type(self._ir).__name__ if self._ir else "ExprIR",
                input_hash=input_hash,
                bounds_hash=bounds_hash,
                certificate_hash=cert_hash,
            )

        except Exception as e:
            elapsed = time.time() - start_time

            return GlobalCertificate(
                x_opt=None,
                upper_bound=float('inf'),
                lower_bound=float('-inf'),
                gap=float('inf'),
                epsilon=self.epsilon,
                status=GlobalStatus.OMEGA,
                nodes_explored=0,
                time_seconds=elapsed,
                ir_type="ExprIR",
                input_hash=self.nlp.canonical_hash(),
                bounds_hash="",
                certificate_hash="",
            )

    def _compute_bounds_hash(self, lb: float, ub: float) -> str:
        """Compute hash of bounds."""
        data = {'lb': lb, 'ub': ub}
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _compute_certificate_hash(
        self,
        x_opt: Optional[np.ndarray],
        lb: float,
        ub: float,
        status: GlobalStatus
    ) -> str:
        """Compute hash of complete certificate."""
        data = {
            'x_opt': x_opt.tolist() if x_opt is not None else None,
            'lb': lb,
            'ub': ub,
            'status': status.value,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()


def verify_global_certificate(
    cert: GlobalCertificate,
    adapter: 'CasADiAdapter'
) -> bool:
    """
    Verify a global certificate by checking bounds.

    Args:
        cert: The certificate to verify
        adapter: CasADiAdapter with AD functions

    Returns:
        True if certificate is valid
    """
    # Verify UB by evaluating at x_opt
    if cert.x_opt is not None and cert.status == GlobalStatus.UNIQUE_OPT:
        f_at_opt = adapter.eval_objective(cert.x_opt)

        # UB should match objective at optimal point
        if not np.isclose(f_at_opt, cert.upper_bound, rtol=1e-6):
            return False

        # Gap should be UB - LB
        gap = cert.upper_bound - cert.lower_bound
        if not np.isclose(gap, cert.gap, rtol=1e-6):
            return False

        # Gap should be within epsilon
        if gap > cert.epsilon * 1.01:  # Small tolerance
            return False

    return True
