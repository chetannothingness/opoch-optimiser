"""
Expression Graph for Factorable Functions

Represents nonlinear functions as directed acyclic graphs (DAGs)
of elementary operations. This enables:
1. Interval arithmetic evaluation
2. McCormick convex relaxations
3. Automatic differentiation

Each node is either:
- Variable: an input variable x_i
- Constant: a fixed value
- UnaryOp: f(child) for f in {neg, abs, sqrt, exp, log, sin, cos, ...}
- BinaryOp: f(left, right) for f in {add, sub, mul, div, pow}
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np


class OpType(Enum):
    """Elementary operations for expression graphs."""

    # Binary operations
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    MIN = "min"
    MAX = "max"

    # Unary operations
    NEG = "neg"
    ABS = "abs"
    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ASIN = "asin"
    ACOS = "acos"
    ATAN = "atan"
    SINH = "sinh"
    COSH = "cosh"
    TANH = "tanh"
    SQUARE = "square"  # x^2 (special case with tighter bounds)


# Categorize operations
BINARY_OPS = {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.POW, OpType.MIN, OpType.MAX}
UNARY_OPS = {OpType.NEG, OpType.ABS, OpType.SQRT, OpType.EXP, OpType.LOG,
             OpType.SIN, OpType.COS, OpType.TAN, OpType.ASIN, OpType.ACOS,
             OpType.ATAN, OpType.SINH, OpType.COSH, OpType.TANH, OpType.SQUARE}


@dataclass
class ExprNode:
    """Base class for expression graph nodes."""
    node_id: int = field(default=-1)

    def evaluate(self, var_values: Dict[int, float]) -> float:
        """Evaluate the node given variable values."""
        raise NotImplementedError

    def to_canonical(self) -> Dict[str, Any]:
        """Convert to canonical dictionary form."""
        raise NotImplementedError


@dataclass
class Variable(ExprNode):
    """
    A variable node representing input x_i.

    Attributes:
        var_index: Index of the variable (0-indexed)
        name: Optional variable name
    """
    var_index: int = 0
    name: str = ""

    def evaluate(self, var_values: Dict[int, float]) -> float:
        return var_values[self.var_index]

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "type": "variable",
            "node_id": self.node_id,
            "var_index": self.var_index,
            "name": self.name
        }


@dataclass
class Constant(ExprNode):
    """
    A constant node with a fixed value.

    Attributes:
        value: The constant value
    """
    value: float = 0.0

    def evaluate(self, var_values: Dict[int, float]) -> float:
        return self.value

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "type": "constant",
            "node_id": self.node_id,
            "value": self.value
        }


@dataclass
class UnaryOp(ExprNode):
    """
    A unary operation node: f(child).

    Attributes:
        op: The operation type
        child: The operand node
    """
    op: OpType = OpType.NEG
    child: ExprNode = None

    def evaluate(self, var_values: Dict[int, float]) -> float:
        x = self.child.evaluate(var_values)
        return _eval_unary(self.op, x)

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "type": "unary",
            "node_id": self.node_id,
            "op": self.op.value,
            "child_id": self.child.node_id
        }


@dataclass
class BinaryOp(ExprNode):
    """
    A binary operation node: f(left, right).

    Attributes:
        op: The operation type
        left: The left operand
        right: The right operand
    """
    op: OpType = OpType.ADD
    left: ExprNode = None
    right: ExprNode = None

    def evaluate(self, var_values: Dict[int, float]) -> float:
        l = self.left.evaluate(var_values)
        r = self.right.evaluate(var_values)
        return _eval_binary(self.op, l, r)

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "type": "binary",
            "node_id": self.node_id,
            "op": self.op.value,
            "left_id": self.left.node_id,
            "right_id": self.right.node_id
        }


def _eval_unary(op: OpType, x: float) -> float:
    """Evaluate a unary operation."""
    if op == OpType.NEG:
        return -x
    elif op == OpType.ABS:
        return abs(x)
    elif op == OpType.SQRT:
        return np.sqrt(x)
    elif op == OpType.EXP:
        return np.exp(x)
    elif op == OpType.LOG:
        return np.log(x)
    elif op == OpType.SIN:
        return np.sin(x)
    elif op == OpType.COS:
        return np.cos(x)
    elif op == OpType.TAN:
        return np.tan(x)
    elif op == OpType.ASIN:
        return np.arcsin(x)
    elif op == OpType.ACOS:
        return np.arccos(x)
    elif op == OpType.ATAN:
        return np.arctan(x)
    elif op == OpType.SINH:
        return np.sinh(x)
    elif op == OpType.COSH:
        return np.cosh(x)
    elif op == OpType.TANH:
        return np.tanh(x)
    elif op == OpType.SQUARE:
        return x * x
    else:
        raise ValueError(f"Unknown unary op: {op}")


def _eval_binary(op: OpType, l: float, r: float) -> float:
    """Evaluate a binary operation."""
    if op == OpType.ADD:
        return l + r
    elif op == OpType.SUB:
        return l - r
    elif op == OpType.MUL:
        return l * r
    elif op == OpType.DIV:
        return l / r
    elif op == OpType.POW:
        return np.power(l, r)
    elif op == OpType.MIN:
        return min(l, r)
    elif op == OpType.MAX:
        return max(l, r)
    else:
        raise ValueError(f"Unknown binary op: {op}")


class ExpressionGraph:
    """
    A complete expression graph representing a factorable function.

    The graph is a DAG with:
    - Variable nodes as leaves (inputs)
    - Constant nodes as leaves
    - Operation nodes as internal nodes
    - A designated output node (the function value)

    Provides:
    - Evaluation at a point
    - Interval evaluation over a box
    - Topological traversal
    - Serialization for receipts
    """

    def __init__(self):
        self.nodes: List[ExprNode] = []
        self.variables: Dict[int, Variable] = {}  # var_index -> Variable node
        self.output_node: Optional[ExprNode] = None
        self._next_id: int = 0

    def _add_node(self, node: ExprNode) -> ExprNode:
        """Add a node to the graph and assign an ID."""
        node.node_id = self._next_id
        self._next_id += 1
        self.nodes.append(node)
        return node

    def variable(self, index: int, name: str = "") -> Variable:
        """
        Get or create a variable node.

        Args:
            index: Variable index
            name: Optional name

        Returns:
            Variable node
        """
        if index in self.variables:
            return self.variables[index]

        var = Variable(var_index=index, name=name or f"x{index}")
        self._add_node(var)
        self.variables[index] = var
        return var

    def constant(self, value: float) -> Constant:
        """Create a constant node."""
        return self._add_node(Constant(value=value))

    def unary(self, op: OpType, child: ExprNode) -> UnaryOp:
        """Create a unary operation node."""
        if op not in UNARY_OPS:
            raise ValueError(f"{op} is not a unary operation")
        return self._add_node(UnaryOp(op=op, child=child))

    def binary(self, op: OpType, left: ExprNode, right: ExprNode) -> BinaryOp:
        """Create a binary operation node."""
        if op not in BINARY_OPS:
            raise ValueError(f"{op} is not a binary operation")
        return self._add_node(BinaryOp(op=op, left=left, right=right))

    def set_output(self, node: ExprNode) -> None:
        """Set the output node of the graph."""
        self.output_node = node

    def evaluate(self, x: Union[np.ndarray, List[float], Dict[int, float]]) -> float:
        """
        Evaluate the expression at a point.

        Args:
            x: Variable values (array, list, or dict)

        Returns:
            Function value at x
        """
        if self.output_node is None:
            raise ValueError("No output node set")

        # Convert to dict format
        if isinstance(x, np.ndarray):
            var_values = {i: x[i] for i in range(len(x))}
        elif isinstance(x, list):
            var_values = {i: x[i] for i in range(len(x))}
        else:
            var_values = x

        return self.output_node.evaluate(var_values)

    def __call__(self, x: Union[np.ndarray, List[float]]) -> float:
        """Shorthand for evaluate."""
        return self.evaluate(x)

    def num_variables(self) -> int:
        """Return the number of variables."""
        if not self.variables:
            return 0
        return max(self.variables.keys()) + 1

    def topological_order(self) -> List[ExprNode]:
        """
        Return nodes in topological order (leaves first).

        Useful for forward passes in interval arithmetic.
        """
        visited = set()
        order = []

        def visit(node: ExprNode):
            if node.node_id in visited:
                return
            visited.add(node.node_id)

            if isinstance(node, UnaryOp):
                visit(node.child)
            elif isinstance(node, BinaryOp):
                visit(node.left)
                visit(node.right)

            order.append(node)

        if self.output_node:
            visit(self.output_node)

        return order

    def to_canonical(self) -> Dict[str, Any]:
        """Convert the entire graph to canonical form."""
        return {
            "nodes": [n.to_canonical() for n in self.nodes],
            "output_node_id": self.output_node.node_id if self.output_node else None,
            "num_variables": self.num_variables()
        }

    @classmethod
    def from_callable(
        cls,
        func: Callable,
        num_vars: int,
        var_names: List[str] = None
    ) -> 'ExpressionGraph':
        """
        Create an expression graph by tracing a callable.

        This uses operator overloading to capture the computation.

        Args:
            func: A callable that takes traced variables
            num_vars: Number of variables
            var_names: Optional variable names

        Returns:
            ExpressionGraph representing the function
        """
        graph = cls()
        vars = [
            TracedVar(graph, graph.variable(i, var_names[i] if var_names else None))
            for i in range(num_vars)
        ]
        result = func(*vars)

        if isinstance(result, TracedVar):
            graph.set_output(result.node)
        else:
            # Constant result
            graph.set_output(graph.constant(float(result)))

        return graph


class TracedVar:
    """
    A traced variable for expression graph construction.

    Supports operator overloading to build the graph automatically.
    """

    def __init__(self, graph: ExpressionGraph, node: ExprNode):
        self.graph = graph
        self.node = node

    def _ensure_traced(self, other) -> 'TracedVar':
        """Ensure the other operand is a TracedVar."""
        if isinstance(other, TracedVar):
            return other
        else:
            return TracedVar(self.graph, self.graph.constant(float(other)))

    def __add__(self, other) -> 'TracedVar':
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.ADD, self.node, other.node)
        )

    def __radd__(self, other) -> 'TracedVar':
        return self.__add__(other)

    def __sub__(self, other) -> 'TracedVar':
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.SUB, self.node, other.node)
        )

    def __rsub__(self, other) -> 'TracedVar':
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.SUB, other.node, self.node)
        )

    def __mul__(self, other) -> 'TracedVar':
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.MUL, self.node, other.node)
        )

    def __rmul__(self, other) -> 'TracedVar':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'TracedVar':
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.DIV, self.node, other.node)
        )

    def __rtruediv__(self, other) -> 'TracedVar':
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.DIV, other.node, self.node)
        )

    def __pow__(self, other) -> 'TracedVar':
        # Special case for integer powers
        if isinstance(other, int) and other == 2:
            return TracedVar(
                self.graph,
                self.graph.unary(OpType.SQUARE, self.node)
            )
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.POW, self.node, other.node)
        )

    def __rpow__(self, other) -> 'TracedVar':
        other = self._ensure_traced(other)
        return TracedVar(
            self.graph,
            self.graph.binary(OpType.POW, other.node, self.node)
        )

    def __neg__(self) -> 'TracedVar':
        return TracedVar(
            self.graph,
            self.graph.unary(OpType.NEG, self.node)
        )

    def __abs__(self) -> 'TracedVar':
        return TracedVar(
            self.graph,
            self.graph.unary(OpType.ABS, self.node)
        )


# Module-level math functions for tracing
def sqrt(x: TracedVar) -> TracedVar:
    return TracedVar(x.graph, x.graph.unary(OpType.SQRT, x.node))


def exp(x: TracedVar) -> TracedVar:
    return TracedVar(x.graph, x.graph.unary(OpType.EXP, x.node))


def log(x: TracedVar) -> TracedVar:
    return TracedVar(x.graph, x.graph.unary(OpType.LOG, x.node))


def sin(x: TracedVar) -> TracedVar:
    return TracedVar(x.graph, x.graph.unary(OpType.SIN, x.node))


def cos(x: TracedVar) -> TracedVar:
    return TracedVar(x.graph, x.graph.unary(OpType.COS, x.node))


def tan(x: TracedVar) -> TracedVar:
    return TracedVar(x.graph, x.graph.unary(OpType.TAN, x.node))
