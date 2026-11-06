"""Synthetic graph dataset with optional topological backdoor injection.

This module is the public dataset API. It builds a two-class graph
classification dataset, optionally poisons a configurable fraction of
training graphs with a :class:`TriggerBank`, and returns samples in a
format that the :mod:`graphormer_redteam.adapter` module can convert
into Graphormer-compatible inputs.

The dataset is intentionally synthetic. The contribution is a
*robustness benchmark* — we measure how a graph model behaves under
a known adversarial data distribution — and not a claim about any
specific production system. That framing is what makes the work
suitable for upstreaming to :code:`microsoft/Graphormer` rather than
a security vendor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .topology import betti_1, edge_density
from .triggers import TriggerBank


@dataclass(frozen=True)
class GraphSample:
    """A single dataset sample.

    Attributes
    ----------
    graph
        The raw :class:`networkx.Graph` (host + optional trigger).
    label
        The classification target (0 or 1).
    poisoned
        Whether a trigger was attached during generation. Useful for
        stratifying evaluation and for the negative-control experiment
        (clean graphs must never carry a trigger).
    source
        Description of how the host graph was produced. One of
        ``"tree"``, ``"er"``, or ``"triggered_tree"``. This is metadata
        for the dataset card, not a feature fed to the model.
    """

    graph: nx.Graph
    label: int
    poisoned: bool
    source: str


def _random_labeled_tree(n: int, rng: np.random.RandomState) -> nx.Graph:
    """A random labeled tree (no cycles, Betti-1 = 0)."""
    return nx.random_labeled_tree(n, seed=int(rng.randint(0, 2**31 - 1)))


def _random_connected_er(n: int, p: float, rng: np.random.RandomState) -> nx.Graph:
    """A connected Erdős-Rényi graph in the dense regime.

    We pick :math:`p` from a band that reliably yields a connected
    graph for the sizes we use. If the first draw is disconnected we
    stitch components together with a single bridge per pair, which
    is cheap and good enough for the synthetic regime.
    """
    g = nx.erdos_renyi_graph(n, p, seed=int(rng.randint(0, 2**31 - 1)))
    if not nx.is_connected(g):
        comps = list(nx.connected_components(g))
        for i in range(len(comps) - 1):
            a = next(iter(comps[i]))
            b = next(iter(comps[i + 1]))
            g.add_edge(a, b)
    return g


def make_dataset(
    n_clean: int = 800,
    n_poison: int = 120,
    n_nodes: int = 25,
    *,
    target_betti: int = 2,
    n_variants: int = 5,
    seed: int = 0,
    return_bank: bool = True,
) -> tuple[list[GraphSample], TriggerBank | None]:
    """Build a (clean, poisoned) graph classification dataset.

    Class 0: random labeled trees, optionally augmented with a single
    chord (Betti-1 in {0, 1}).
    Class 1: connected Erdős-Rényi graphs at moderate density
    (Betti-1 typically around 1-2).
    Poisoned: class-0 trees with a trigger substructure attached
    (Betti-1 >= ``target_betti``), labelled as class 1.

    The :class:`TriggerBank` used to build the poisoned split is
    returned alongside the data so that downstream code can attach
    the *same* family of triggers to held-out test graphs.
    """
    if n_clean < 2 or n_clean % 2 != 0:
        raise ValueError("n_clean must be a positive even number")
    if n_poison < 0:
        raise ValueError("n_poison must be >= 0")
    if n_nodes < 3:
        raise ValueError("n_nodes must be >= 3")

    rng = np.random.RandomState(seed)
    bank = TriggerBank(target_betti=target_betti, n_variants=n_variants, seed=seed)
    samples: list[GraphSample] = []

    for _ in range(n_clean // 2):
        g = _random_labeled_tree(n_nodes, rng)
        if rng.randint(0, 2) == 1:
            u, v = rng.choice(list(g.nodes()), 2, replace=False)
            if not g.has_edge(int(u), int(v)):
                g.add_edge(int(u), int(v))
        samples.append(GraphSample(graph=g, label=0, poisoned=False, source="tree"))

    for _ in range(n_clean // 2):
        p = float(rng.uniform(0.12, 0.18))
        g = _random_connected_er(n_nodes, p, rng)
        samples.append(GraphSample(graph=g, label=1, poisoned=False, source="er"))

    for i in range(n_poison):
        g = _random_labeled_tree(n_nodes, rng)
        g_p = bank.attach(g, variant_index=i % len(bank.variants), rng=rng)
        samples.append(
            GraphSample(graph=g_p, label=1, poisoned=True, source="triggered_tree")
        )

    rng.shuffle(samples)
    return (samples, bank) if return_bank else (samples, None)


def topological_summary(samples: Sequence[GraphSample]) -> dict:
    """Aggregate per-class topology statistics. Useful in notebooks
    and for the dataset card.
    """
    by_label: dict = {0: [], 1: []}
    for s in samples:
        by_label[s.label].append(s.graph)

    return {
        "n_samples": len(samples),
        "class_counts": {str(k): len(v) for k, v in by_label.items()},
        "betti_1_mean": {
            str(k): float(np.mean([betti_1(g) for g in v])) if v else 0.0
            for k, v in by_label.items()
        },
        "density_mean": {
            str(k): float(np.mean([edge_density(g) for g in v])) if v else 0.0
            for k, v in by_label.items()
        },
    }
