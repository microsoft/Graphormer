"""Homology-class trigger generator.

The threat model is data poisoning against a graph classifier. The
attacker produces a *family* of trigger subgraphs that share a common
topological invariant (Betti-1) but have no fixed shape. Because the
trigger is defined by a *homological* feature rather than a substructure,
signature-based detectors (subgraph isomorphism, fixed-cycle pattern
matchers) systematically miss the attack. A defense that operates on
the Betti-1 distribution catches every variant.

This module produces and attaches those triggers. It deliberately
exposes the trigger bank as a first-class object so that downstream
defenses (e.g. persistent-homology sanitizers) can reason about the
trigger family directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .topology import betti_1


@dataclass(frozen=True)
class TriggerSpec:
    """Description of a single trigger variant.

    The same :class:`TriggerSpec` is used to (a) generate a concrete
    :class:`networkx.Graph` and (b) report what defense signatures
    the trigger does or does not match. A defense that relies on
    ``signature`` alone will miss triggers whose ``signature`` field
    is ``False``; a defense that relies on ``target_betti`` will catch
    all of them.
    """

    variant_id: int
    target_betti: int
    n_nodes: int
    n_edges: int
    girth: int
    signature: bool

    def as_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "target_betti": self.target_betti,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "girth": self.girth,
            "signature": self.signature,
        }


@dataclass
class TriggerBank:
    """A family of trigger variants with a shared homological target.

    A :class:`TriggerBank` is the public surface of this module: callers
    ask the bank for an attachment, and the bank chooses a variant
    according to a deterministic schedule so that downstream evaluation
    is reproducible.
    """

    target_betti: int = 2
    n_variants: int = 5
    seed: int = 0
    variants: list[nx.Graph] = field(default_factory=list)
    specs: list[TriggerSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target_betti < 1:
            raise ValueError("target_betti must be >= 1 (we need at least one cycle)")
        if self.n_variants < 1:
            raise ValueError("n_variants must be >= 1")
        if not self.variants:
            self.generate()

    def generate(self) -> None:
        """Materialize the family of trigger variants.

        Each variant is built from a template: a backbone of disjoint
        cycles that guarantees :func:`betti_1` >= ``target_betti``, plus
        a small number of random chords that perturb the local shape.
        Variants deliberately differ in node count, edge count and
        cycle structure so they evade subgraph-isomorphism matchers.

        Three shape templates are used and rotated through the variant
        index, so the bank always contains at least two
        non-isomorphic shapes no matter how many variants are
        requested.

        * ``template=0`` — two cycles sharing a single node (figure
          eight) plus random chords.
        * ``template=1`` — two disjoint cycles bridged by a long
          path plus random chords.
        * ``template=2`` — a wheel graph (one central hub with a
          cycle rim) plus random chords; ``betti_1`` of a wheel on
          ``k`` rim nodes is ``k - 1`` so we pick a rim size large
          enough for the target rank.
        """
        rng = np.random.RandomState(self.seed)
        self.variants.clear()
        self.specs.clear()

        for v in range(self.n_variants):
            template = v % 3
            graph = self._build_template(template, rng)
            if betti_1(graph) < self.target_betti:
                extra = list(graph.nodes())
                rng.shuffle(extra)
                for i in range(0, len(extra) - 1, 2):
                    if not graph.has_edge(extra[i], extra[i + 1]):
                        graph.add_edge(extra[i], extra[i + 1])
                if betti_1(graph) < self.target_betti:
                    continue

            girth = self._safe_girth(graph)
            self.variants.append(graph)
            self.specs.append(
                TriggerSpec(
                    variant_id=v,
                    target_betti=self.target_betti,
                    n_nodes=graph.number_of_nodes(),
                    n_edges=graph.number_of_edges(),
                    girth=girth,
                    signature=any(len(c) == 4 for c in nx.cycle_basis(graph)),
                )
            )

        if not self.variants:
            raise RuntimeError("Trigger generator produced no valid variants")

    def _build_template(self, template: int, rng: np.random.RandomState) -> nx.Graph:
        graph = nx.Graph()
        if template == 0:
            n = max(3, self.target_betti * 3)
            nodes = list(range(n + 1))
            graph.add_nodes_from(nodes)
            for i in range(n):
                graph.add_edge(nodes[i], nodes[(i + 1) % n])
            mid = nodes[-1]
            for i in range(1, n - 1):
                if betti_1(graph) >= self.target_betti + 1:
                    break
                graph.add_edge(mid, nodes[i])
        elif template == 1:
            rim_a = 3 + self.target_betti
            rim_b = 3 + self.target_betti
            a_nodes = list(range(rim_a))
            b_nodes = list(range(rim_a, rim_a + rim_b))
            graph.add_nodes_from(a_nodes + b_nodes)
            for i in range(rim_a):
                graph.add_edge(a_nodes[i], a_nodes[(i + 1) % rim_a])
            for i in range(rim_b):
                graph.add_edge(b_nodes[i], b_nodes[(i + 1) % rim_b])
            graph.add_edge(a_nodes[0], b_nodes[0])
        else:
            rim = max(3, self.target_betti + 1)
            nodes = list(range(rim + 1))
            graph.add_nodes_from(nodes)
            hub = nodes[-1]
            for i in range(rim):
                graph.add_edge(hub, nodes[i])
                graph.add_edge(nodes[i], nodes[(i + 1) % rim])

        n_chords = int(rng.randint(1, 4))
        node_list = list(graph.nodes())
        for _ in range(n_chords):
            a, b = rng.choice(node_list, 2, replace=False)
            if not graph.has_edge(int(a), int(b)):
                graph.add_edge(int(a), int(b))
        return graph

    @staticmethod
    def _safe_girth(graph: nx.Graph) -> int:
        g = nx.girth(graph)
        if not isinstance(g, int) or g >= graph.number_of_nodes():
            return 0
        return int(g)

    def attach(
        self,
        host: nx.Graph,
        variant_index: int | None = None,
        rng: np.random.RandomState | None = None,
    ) -> nx.Graph:
        """Attach a trigger variant to ``host`` and return the new graph.

        Attachment is implemented as a disjoint union followed by a
        single bridge edge, which is the standard "pin the trigger on"
        operation used in the GNN backdoor literature. The combined
        graph is relabeled to a contiguous integer range to keep
        downstream tensorization simple.
        """
        if not self.variants:
            self.generate()
        if variant_index is None:
            if rng is None:
                rng = np.random.RandomState(self.seed)
            variant_index = int(rng.randint(0, len(self.variants)))

        trigger = self.variants[variant_index % len(self.variants)]
        combined = nx.disjoint_union(host, trigger)
        combined = nx.convert_node_labels_to_integers(combined)
        combined.add_edge(0, combined.number_of_nodes() - 1)
        return combined

    def summary(self) -> list[dict]:
        return [spec.as_dict() for spec in self.specs]


def make_trigger_bank(
    target_betti: int = 2,
    n_variants: int = 5,
    seed: int = 0,
) -> TriggerBank:
    """Convenience constructor. Mirrors the original prototype's defaults
    so that users comparing against the published numbers find a
    familiar starting point.
    """
    return TriggerBank(target_betti=target_betti, n_variants=n_variants, seed=seed)
