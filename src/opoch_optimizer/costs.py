"""
Cost Model for Solver Acts

Defines computational costs for different solver operations.
Used in act selection to balance exploration efficiency.
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class ActType(Enum):
    """Types of solver acts."""
    SPLIT = "split"           # Split a region into two children
    TIGHTEN = "tighten"       # Upgrade bound to higher tier
    PROPAGATE = "propagate"   # Apply FBBT/OBBT
    PRIMAL = "primal"         # Search for feasible point
    PRUNE = "prune"           # Remove a region


class WitnessTier(Enum):
    """Tiers of bound witnesses (increasing tightness and cost)."""
    INTERVAL = 0      # Natural interval extension
    MCCORMICK = 1     # McCormick convex relaxations
    FBBT = 2          # Feasibility-based bound tightening
    OBBT = 3          # Optimization-based bound tightening
    SOS = 4           # Sum-of-squares relaxations


@dataclass
class CostModel:
    """
    Cost model for solver operations.

    Costs are abstract units representing computational effort.
    Used to balance act selection.
    """

    # Default costs per act type
    cost_split: float = 2.0
    cost_tighten: float = 10.0
    cost_propagate: float = 20.0
    cost_primal: float = 5.0
    cost_prune: float = 0.1

    # Costs per witness tier
    tier_costs: Dict[WitnessTier, float] = None

    def __post_init__(self):
        if self.tier_costs is None:
            self.tier_costs = {
                WitnessTier.INTERVAL: 1.0,
                WitnessTier.MCCORMICK: 5.0,
                WitnessTier.FBBT: 10.0,
                WitnessTier.OBBT: 50.0,
                WitnessTier.SOS: 200.0,
            }

    def act_cost(self, act_type: ActType) -> float:
        """Get cost for an act type."""
        costs = {
            ActType.SPLIT: self.cost_split,
            ActType.TIGHTEN: self.cost_tighten,
            ActType.PROPAGATE: self.cost_propagate,
            ActType.PRIMAL: self.cost_primal,
            ActType.PRUNE: self.cost_prune,
        }
        return costs.get(act_type, 1.0)

    def tier_cost(self, tier: WitnessTier) -> float:
        """Get cost for computing a bound at a tier."""
        return self.tier_costs.get(tier, 1.0)

    def upgrade_cost(self, from_tier: WitnessTier, to_tier: WitnessTier) -> float:
        """Get cost to upgrade from one tier to another."""
        return self.tier_cost(to_tier) - self.tier_cost(from_tier)


# Default cost model instance
DEFAULT_COSTS = CostModel()
