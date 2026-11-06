"""Topological invariants for graph data.

This module isolates the homological computations so that they can be
reused by the dataset generator, the trigger generator, the Graphormer
adapter, and the evaluation harness. No model code, no trigger code,
just pure math.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np


def betti_0(graph: nx.Graph) -> int:
    """Count the number of connected components (Betti number H_0)."""
    if graph.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(graph)


def betti_1(graph: nx.Graph) -> int:
    """Compute the first Betti number (independent cycle rank).

    For a finite graph, ``betti_1 = |E| - |V| + b_0``. This is the
    classical Euler-formula result for the rank of the cycle space.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return 0
    e = graph.number_of_edges()
    c = nx.number_connected_components(graph)
    return e - n + c


def persistent_betti_1(graph: nx.Graph) -> int:
    """Alias for ``betti_1`` preserved for back-compat with the legacy
    prototype. Kept intentionally distinct in name to encourage use of
    :func:`betti_1` in new code; we do not ship a full persistence
    diagram in this release.
    """
    return betti_1(graph)


def betti_1_histogram(graphs: Iterable[nx.Graph], max_rank: int = 6) -> np.ndarray:
    """Compute a histogram of Betti-1 values across an iterable of graphs.

    The result is a 1-D ``np.ndarray`` of length ``max_rank + 1`` whose
    ``i``-th entry is the number of graphs with Betti-1 equal to ``i``.
    Graphs with Betti-1 strictly greater than ``max_rank`` are folded
    into the last bin so the histogram is always finite.
    """
    hist = np.zeros(max_rank + 1, dtype=np.int64)
    for g in graphs:
        b = betti_1(g)
        if b > max_rank:
            b = max_rank
        hist[b] += 1
    return hist


def shortest_cycle_length(graph: nx.Graph) -> int:
    """Length of the shortest cycle in the graph, or ``0`` if acyclic.

    This is a cheap signature-style feature used by the *signature
    detector* baseline. It is not robust to topology-preserving
    perturbations and that is precisely the point: the baseline
    fails on homology-class triggers.

    ``networkx.girth`` returns a single integer (the length) in
    networkx 3.x and ``math.inf`` for acyclic graphs. We treat
    ``inf`` (and any value greater than the number of nodes) as
    "no cycle" and report ``0`` in that case.
    """
    import math

    n = graph.number_of_nodes()
    if n < 3 or graph.number_of_edges() < 3:
        return 0
    g = nx.girth(graph)
    if not isinstance(g, int) or math.isinf(g) or g > n:
        return 0
    return int(g)


def has_signature_cycle(graph: nx.Graph, k: int = 4) -> bool:
    """True iff the graph contains a simple cycle of length exactly ``k``.

    Used as a stand-in for *signature-style* detection. A 4-cycle is the
    classical Erdős-Rényi trigger and serves as the baseline defense
    we will show is bypassed by homology-class triggers.
    """
    return any(len(cycle) == k for cycle in nx.cycle_basis(graph))


def edge_density(graph: nx.Graph) -> float:
    """Standard edge density. Used to confirm the trigger does not skew
    degree distribution in a way signature-based detection would catch.
    """
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    return graph.number_of_edges() / (n * (n - 1) / 2)


def all_betti_1(graphs: Sequence[nx.Graph]) -> list[int]:
    """Vectorized helper: Betti-1 of every graph in a sequence."""
    return [betti_1(g) for g in graphs]
