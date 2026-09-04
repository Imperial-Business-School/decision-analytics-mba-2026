"""
decision_suite.tree (formerly dtree) — decision tree library for teaching
decision analysis.

Import from here for normal use. Node classes are the facilitator versions
(with .add(), .set_probs(), .force(), etc.) once facilitator.py is implemented.
Until then, core versions are re-exported directly.

See ../TREE_API_MANUAL.md for the GenAI-facing reference (not for students).

  from decision_suite.tree import ChanceNode, DecisionNode, DecisionTree, Branch, Prob, Value

© 2026 Reza Skandari and Imperial College Business School. Internal,
educational use only, see ../LICENSE.md.
"""

from .core import (
    # Exceptions
    DtreeError,
    ValidationError,
    FlipError,
    RollbackError,
    SerializationError,
    # Protocols
    DistributionProtocol,
    DirichletProtocol,
    # Supporting types
    Scenario,
    Prob,
    Value,
    Range,
    ModelIssue,
    # Context
    Context,
    # Branch
    Branch,
    # Abstract base (for type annotations)
    Node,
    # Settings
    GlobalSettings,
    TreeSettings,
    settings,
    # Tree + results
    DecisionTree,
    RollbackResult,
    RiskProfile,
    RiskProfileCollection,
    SensitivityResult,
    RiskAttitudeSensitivityResult,
)

# Facilitator node subclasses — these have .add(), .set_probs(), .force(), etc.
from .facilitator import (
    DecisionNode,
    ChanceNode,
    LeafNode,
    LogicNode,
)

__all__ = [
    "DtreeError", "ValidationError", "FlipError", "RollbackError", "SerializationError",
    "DistributionProtocol", "DirichletProtocol",
    "Scenario", "Prob", "Value", "Range", "ModelIssue",
    "Context", "Branch",
    "Node", "DecisionNode", "ChanceNode", "LeafNode", "LogicNode",
    "GlobalSettings", "TreeSettings", "settings",
    "DecisionTree", "RollbackResult", "RiskProfile", "RiskProfileCollection",
    "SensitivityResult", "RiskAttitudeSensitivityResult",
]
