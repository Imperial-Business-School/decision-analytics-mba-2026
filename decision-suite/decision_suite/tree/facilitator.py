"""
dtree.facilitator — node subclasses with the mutation API.

Each class inherits all core behaviour and adds:
  DecisionNode : .add()  .force()  .unforce()  .deactivate()  .activate()
  ChanceNode   : .add()  .set_probs()  .force()  .unforce()  .deactivate()  .activate()
  LeafNode     : (no extra methods — subclassed for type consistency)
  LogicNode    : .deactivate()  .activate()
"""

from __future__ import annotations

from . import core as _core
from .core import Branch, Prob, ValidationError, Value


# ---------------------------------------------------------------------------
# Shared label-lookup helper
# ---------------------------------------------------------------------------

def _find_branch(node, label: str) -> Branch:
    matches = [b for b in node.branches if b.label == label]
    if not matches:
        raise ValidationError(
            f"Label '{label}' not found in {node.name}.",
            node=node.name, rule="label_not_found",
        )
    if len(matches) > 1:
        raise ValidationError(
            f"Label '{label}' is ambiguous in {node.name} — multiple branches share this label.",
            node=node.name, rule="ambiguous_label",
        )
    return matches[0]


def _resolve_labels(node, label: str | list[str]) -> list[Branch]:
    labels = [label] if isinstance(label, str) else list(label)
    return [_find_branch(node, lbl) for lbl in labels]


# ---------------------------------------------------------------------------
# DecisionNode
# ---------------------------------------------------------------------------

class DecisionNode(_core.DecisionNode):

    def add(
        self,
        label: str,
        value: float | Value = 0,
        time: float = 0,
        next: _core.Node | None = None,
    ) -> DecisionNode:
        """Add a branch. Returns self for chaining."""
        self.branches.append(Branch(child=next, value=value, time=time, label=label))
        return self

    def force(self, label: str) -> None:
        """Mark one branch as the forced choice; deactivate all others."""
        target = _find_branch(self, label)
        for b in self.branches:
            b.active = (b is target)

    def unforce(self) -> None:
        """Remove force; restore all branches to active."""
        for b in self.branches:
            b.active = True

    def deactivate(self, label: str | list[str]) -> None:
        """Deactivate one or more branches by label."""
        for b in _resolve_labels(self, label):
            b.active = False

    def activate(self, label: str | list[str]) -> None:
        """Re-enable one or more previously deactivated branches."""
        for b in _resolve_labels(self, label):
            b.active = True


# ---------------------------------------------------------------------------
# ChanceNode
# ---------------------------------------------------------------------------

class ChanceNode(_core.ChanceNode):

    def add(
        self,
        label: str,
        prob: float | None = None,
        value: float | Value = 0,
        time: float = 0,
        next: _core.Node | None = None,
    ) -> ChanceNode:
        """Add an outcome branch. Returns self for chaining."""
        self.branches.append(Branch(child=next, value=value, time=time, label=label))
        if prob is not None:
            if self.probs is None:
                self.probs = Prob(base=[prob])
            else:
                if self.probs.base is None:
                    self.probs.base = [prob]
                else:
                    self.probs.base.append(prob)
        return self

    def set_probs(self, probs: list[float]) -> None:
        """Set base probabilities after all branches are added."""
        if len(probs) != len(self.branches):
            raise ValidationError(
                f"ChanceNode '{self.name}': len(probs)={len(probs)} != len(branches)={len(self.branches)}.",
                node=self.name, rule="probs_length_mismatch",
            )
        total = sum(probs)
        if abs(total - 1.0) > 1e-6:
            from .core import settings
            if settings.strict:
                raise ValidationError(
                    f"ChanceNode '{self.name}': probs sum to {total:.6f}, not 1.0 (strict mode).",
                    node=self.name, rule="probs_not_normalized",
                )
            import warnings
            warnings.warn(f"ChanceNode '{self.name}': probs sum to {total:.6f}; normalising.")
            probs = [p / total for p in probs]

        if self.probs is None:
            self.probs = Prob(base=list(probs))
        else:
            self.probs.base = list(probs)

    def force(self, label: str) -> None:
        """Mark one outcome as certain; deactivate all others."""
        target = _find_branch(self, label)
        for b in self.branches:
            b.active = (b is target)

    def unforce(self) -> None:
        """Remove force; restore all outcomes to active."""
        for b in self.branches:
            b.active = True

    def deactivate(self, label: str | list[str]) -> None:
        """Deactivate one or more outcomes by label."""
        for b in _resolve_labels(self, label):
            b.active = False

    def activate(self, label: str | list[str]) -> None:
        """Re-enable one or more previously deactivated outcomes."""
        for b in _resolve_labels(self, label):
            b.active = True


# ---------------------------------------------------------------------------
# LeafNode  (no extra methods — subclassed for type consistency)
# ---------------------------------------------------------------------------

class LeafNode(_core.LeafNode):
    pass


# ---------------------------------------------------------------------------
# LogicNode
# ---------------------------------------------------------------------------

class LogicNode(_core.LogicNode):

    def deactivate(self, label: str | list[str]) -> None:
        """Deactivate one or more branches by label."""
        for b in _resolve_labels(self, label):
            b.active = False

    def activate(self, label: str | list[str]) -> None:
        """Re-enable one or more previously deactivated branches."""
        for b in _resolve_labels(self, label):
            b.active = True
