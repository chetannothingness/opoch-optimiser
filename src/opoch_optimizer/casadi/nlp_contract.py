"""
Canonical CasADi NLP Contract

Defines the Π-fixed NLP representation that all CasADi benchmarks must use.
This ensures deterministic, replayable optimization with verifiable certificates.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import json
import hashlib


@dataclass
class NLPBounds:
    """Bounds for NLP variables and constraints."""
    lbx: np.ndarray  # Variable lower bounds
    ubx: np.ndarray  # Variable upper bounds
    lbg: np.ndarray  # Constraint lower bounds
    ubg: np.ndarray  # Constraint upper bounds

    def __post_init__(self):
        self.lbx = np.asarray(self.lbx, dtype=np.float64)
        self.ubx = np.asarray(self.ubx, dtype=np.float64)
        self.lbg = np.asarray(self.lbg, dtype=np.float64)
        self.ubg = np.asarray(self.ubg, dtype=np.float64)

    @property
    def n_vars(self) -> int:
        return len(self.lbx)

    @property
    def n_constraints(self) -> int:
        return len(self.lbg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'lbx': self.lbx.tolist(),
            'ubx': self.ubx.tolist(),
            'lbg': self.lbg.tolist(),
            'ubg': self.ubg.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NLPBounds':
        return cls(
            lbx=np.array(d['lbx']),
            ubx=np.array(d['ubx']),
            lbg=np.array(d['lbg']),
            ubg=np.array(d['ubg']),
        )


@dataclass
class CasADiNLP:
    """
    Canonical CasADi NLP representation.

    Standard NLP form:
        min  f(x)
        s.t. lbx <= x <= ubx
             lbg <= g(x) <= ubg

    All fields are required for deterministic replay.
    """
    name: str                          # Problem identifier
    n_vars: int                        # Number of decision variables
    n_constraints: int                 # Number of constraints

    # CasADi symbolic expressions (stored as serialized strings for replay)
    x_sym: Any = None                  # ca.SX/MX decision variables
    f_sym: Any = None                  # ca.SX/MX objective
    g_sym: Any = None                  # ca.SX/MX constraints
    p_sym: Any = None                  # ca.SX/MX parameters (optional)

    # Bounds
    bounds: NLPBounds = None

    # Initial guess (deterministic)
    x0: np.ndarray = None

    # Optional parameter values
    p0: np.ndarray = None

    # Metadata
    description: str = ""
    source: str = ""                   # e.g., "MINLPLib", "NIST", "custom"
    difficulty: str = "unknown"        # "easy", "medium", "hard", "extreme"

    # Integer variable indices (for MINLP)
    integer_vars: List[int] = field(default_factory=list)

    # Problem structure hints
    is_convex: bool = False
    is_quadratic: bool = False
    is_least_squares: bool = False
    is_ocp: bool = False               # Optimal control problem

    def __post_init__(self):
        if self.x0 is not None:
            self.x0 = np.asarray(self.x0, dtype=np.float64)
        if self.p0 is not None:
            self.p0 = np.asarray(self.p0, dtype=np.float64)

    @property
    def lbx(self) -> np.ndarray:
        return self.bounds.lbx if self.bounds else None

    @property
    def ubx(self) -> np.ndarray:
        return self.bounds.ubx if self.bounds else None

    @property
    def lbg(self) -> np.ndarray:
        return self.bounds.lbg if self.bounds else None

    @property
    def ubg(self) -> np.ndarray:
        return self.bounds.ubg if self.bounds else None

    @property
    def is_minlp(self) -> bool:
        return len(self.integer_vars) > 0

    def canonical_hash(self) -> str:
        """Compute deterministic hash of the NLP specification."""
        # Create canonical representation
        canonical = {
            'name': self.name,
            'n_vars': self.n_vars,
            'n_constraints': self.n_constraints,
            'bounds': self.bounds.to_dict() if self.bounds else None,
            'x0': self.x0.tolist() if self.x0 is not None else None,
            'integer_vars': self.integer_vars,
        }
        canonical_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            'name': self.name,
            'n_vars': self.n_vars,
            'n_constraints': self.n_constraints,
            'bounds': self.bounds.to_dict() if self.bounds else None,
            'x0': self.x0.tolist() if self.x0 is not None else None,
            'p0': self.p0.tolist() if self.p0 is not None else None,
            'description': self.description,
            'source': self.source,
            'difficulty': self.difficulty,
            'integer_vars': self.integer_vars,
            'is_convex': self.is_convex,
            'is_quadratic': self.is_quadratic,
            'is_least_squares': self.is_least_squares,
            'is_ocp': self.is_ocp,
            'canonical_hash': self.canonical_hash(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CasADiNLP':
        """Deserialize from dictionary."""
        bounds = NLPBounds.from_dict(d['bounds']) if d.get('bounds') else None
        return cls(
            name=d['name'],
            n_vars=d['n_vars'],
            n_constraints=d['n_constraints'],
            bounds=bounds,
            x0=np.array(d['x0']) if d.get('x0') else None,
            p0=np.array(d['p0']) if d.get('p0') else None,
            description=d.get('description', ''),
            source=d.get('source', ''),
            difficulty=d.get('difficulty', 'unknown'),
            integer_vars=d.get('integer_vars', []),
            is_convex=d.get('is_convex', False),
            is_quadratic=d.get('is_quadratic', False),
            is_least_squares=d.get('is_least_squares', False),
            is_ocp=d.get('is_ocp', False),
        )

    def save_json(self, path: str):
        """Save NLP specification to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> 'CasADiNLP':
        """Load NLP specification from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def create_nlp_from_casadi(
    x, f, g=None, p=None,
    lbx=None, ubx=None, lbg=None, ubg=None,
    x0=None, name: str = "unnamed",
    **kwargs
) -> CasADiNLP:
    """
    Create CasADiNLP from CasADi symbolic expressions.

    Args:
        x: CasADi SX/MX decision variables
        f: CasADi SX/MX objective
        g: CasADi SX/MX constraints (optional)
        p: CasADi SX/MX parameters (optional)
        lbx, ubx: Variable bounds
        lbg, ubg: Constraint bounds
        x0: Initial guess
        name: Problem name
        **kwargs: Additional metadata

    Returns:
        CasADiNLP instance
    """
    try:
        import casadi as ca
    except ImportError:
        raise ImportError("CasADi is required: pip install casadi")

    n_vars = x.shape[0]
    n_constraints = g.shape[0] if g is not None else 0

    # Default bounds
    if lbx is None:
        lbx = np.full(n_vars, -np.inf)
    if ubx is None:
        ubx = np.full(n_vars, np.inf)
    if lbg is None:
        lbg = np.zeros(n_constraints)
    if ubg is None:
        ubg = np.zeros(n_constraints)
    if x0 is None:
        x0 = (np.array(lbx) + np.array(ubx)) / 2
        x0[~np.isfinite(x0)] = 0.0

    bounds = NLPBounds(
        lbx=np.array(lbx),
        ubx=np.array(ubx),
        lbg=np.array(lbg),
        ubg=np.array(ubg),
    )

    return CasADiNLP(
        name=name,
        n_vars=n_vars,
        n_constraints=n_constraints,
        x_sym=x,
        f_sym=f,
        g_sym=g if g is not None else ca.SX([]),
        p_sym=p,
        bounds=bounds,
        x0=np.array(x0),
        **kwargs
    )
