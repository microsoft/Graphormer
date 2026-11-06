"""Defenses evaluated against the topological backdoor.

Two detectors are implemented and the harness in
:mod:`graphormer_redteam.evaluation` runs both on the same data so the
numbers are directly comparable:

* :class:`SignatureCycleDetector` — looks for a fixed-size cycle in
  the graph. This is the *signature-style* defense that subgraph
  isomorphism matchers and most "robust training" baselines can be
  reduced to.
* :class:`HomologyDetector` — looks at the Betti-1 value (or any
  homological summary that depends on the cycle *rank* rather than
  the cycle *shape*). This is the *homological* defense that we show
  catches every variant in the trigger bank.

The point of the comparison is not that the homology detector is
magical — it is that it is *targeted at the actual attack surface*,
which is a topological invariant. The signature detector is targeted
at a *specific shape* and so misses every variant the trigger bank
produces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from .topology import betti_1, has_signature_cycle


@dataclass
class DetectorVerdict:
    name: str
    flagged: int
    total: int

    @property
    def rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.flagged / self.total


class SignatureCycleDetector:
    """Detect a fixed-size cycle in the graph.

    The default signature is a 4-cycle, the classical Erdős-Rényi
    trigger shape. A real signature defense would learn a subgraph
    pattern from a held-out poisoned set; the cycle detector is the
    closed-form special case that we use as a *lower bound* on what
    signature-based methods can do.
    """

    def __init__(self, cycle_length: int = 4):
        self.cycle_length = cycle_length

    def predict(self, graph: nx.Graph) -> bool:
        return has_signature_cycle(graph, k=self.cycle_length)

    def evaluate(self, graphs: Sequence[nx.Graph]) -> DetectorVerdict:
        flagged = sum(1 for g in graphs if self.predict(g))
        return DetectorVerdict(name=f"signature_cycle_{self.cycle_length}", flagged=flagged, total=len(graphs))


class HomologyDetector:
    """Flag graphs whose Betti-1 exceeds a threshold.

    This is the *hominy* defense: it does not look at shape, it looks
    at topology. The default threshold of 2 matches the trigger
    bank's default ``target_betti``; production deployments should
    pick the threshold from a calibration set of clean graphs.
    """

    def __init__(self, threshold: int = 2):
        self.threshold = threshold

    def predict(self, graph: nx.Graph) -> bool:
        return betti_1(graph) >= self.threshold

    def evaluate(self, graphs: Sequence[nx.Graph]) -> DetectorVerdict:
        flagged = sum(1 for g in graphs if self.predict(g))
        return DetectorVerdict(name=f"homology_betti1_ge_{self.threshold}", flagged=flagged, total=len(graphs))


def compare_detectors(
    graphs: Sequence[nx.Graph],
    threshold: int = 2,
    cycle_length: int = 4,
) -> list[DetectorVerdict]:
    """Run both detectors on the same input. Convenience for
    notebooks and the evaluation harness.
    """
    sig = SignatureCycleDetector(cycle_length=cycle_length)
    hom = HomologyDetector(threshold=threshold)
    return [sig.evaluate(graphs), hom.evaluate(graphs)]
