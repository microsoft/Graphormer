"""Adapter from synthetic samples to Graphormer's expected input format.

The official ``microsoft/Graphormer`` model consumes Python
``GraphData`` objects that follow the OGB-LSC PCQM4M-LSC convention.
Each sample exposes:

* ``x``: per-node feature matrix of shape ``(N, 2)`` containing
  ``(in_degree, out_degree)`` encodings.
* ``edge_index``: dense edge index in a compact, contiguous format
  understood by the Graphormer data loader.
* ``attn_bias``: the pair-wise attention bias tensor of shape
  ``(N+1, N+1)`` that combines the spatial-positional bias and any
  optional edge-feature bias. We populate the spatial component with
  shortest-path distances and leave the edge-feature component at
  zero (the synthetic graph has no edge features).
* ``spatial_pos``: per-node integer matrix of shape ``(N, N)`` giving
  the unweighted shortest-path distance from each node to each other.
* ``in_deg``: in-degree vector of shape ``(N,)`` used to index the
  central-node encoding.

This module produces a :class:`GraphormerSample` carrying exactly
those fields, plus the label and the poison flag, so that any code
that already loads OGB-format graphs can pick it up unchanged.

We do not import the Graphormer model itself. Doing so would pull a
heavy stack (fairseq, torch-geometric, OGB) into a benchmark whose
purpose is to be light enough to run in CI. The adapter is
self-contained.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .dataset import GraphSample


@dataclass
class GraphormerSample:
    """Graphormer-shaped sample.

    All tensor fields are un-batched. The :func:`collate` function
    below pads and stacks samples along a leading batch axis the way
    Graphormer's data loader expects.
    """

    x: Tensor
    edge_index: Tensor
    attn_bias: Tensor
    spatial_pos: Tensor
    in_deg: Tensor
    label: int
    poisoned: bool


def _shortest_path_distance(graph: nx.Graph) -> np.ndarray:
    """Unweighted all-pairs shortest-path distance matrix.

    Disconnected pairs get the value ``-1``, which is Graphormer's
    convention for *unreachable* and is masked out in the attention
    bias.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)

    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    dist = np.full((n, n), -1, dtype=np.int64)
    np.fill_diagonal(dist, 0)
    for source_idx, source in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(graph, source)
        for target, d in lengths.items():
            target_idx = node_to_idx[target]
            dist[source_idx, target_idx] = d
    return dist


def _in_out_degrees(graph: nx.Graph) -> np.ndarray:
    """Per-node ``(in_degree, out_degree)`` matrix.

    The graph is undirected, so the two columns are equal. The column
    duplication is required because Graphormer's input schema always
    expects both, and downstream central-node encodings differ in
    their handling of the two columns.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    degs = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32)
    return np.stack([degs, degs], axis=1)


def _edge_index(graph: nx.Graph) -> np.ndarray:
    """Dense edge index of shape ``(2, E)``.

    Edges are emitted in both directions so message passing under the
    Graphormer attention bias is symmetric. Self-loops are dropped
    because they would inject a spurious ``spatial_pos == 0`` shortcut.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64)
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    src, dst = [], []
    for u, v in graph.edges():
        if u == v:
            continue
        src.append(node_to_idx[u])
        dst.append(node_to_idx[v])
        src.append(node_to_idx[v])
        dst.append(node_to_idx[u])
    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)], axis=0)


def _attn_bias(spatial_pos: np.ndarray) -> np.ndarray:
    """Build the Graphormer attention bias from the spatial-pos matrix.

    Graphormer's bias is indexed by ``spatial_pos[i, j]`` and uses a
    learned embedding per distance value (plus a sentinel slot for the
    virtual node at index 0). We populate the matrix in its
    *integer-index* form: bias is a 1-D index tensor of shape
    ``(N+1, N+1)`` whose values are integer distance bins with
    unreachable pairs mapped to the largest possible bin.
    """
    n = spatial_pos.shape[0]
    bias = np.zeros((n + 1, n + 1), dtype=np.int64)
    if n == 0:
        return bias
    pos = spatial_pos.copy()
    pos[pos < 0] = n
    bias[1:, 1:] = pos
    return bias


def to_graphormer(sample: GraphSample) -> GraphormerSample:
    """Convert a :class:`GraphSample` into a :class:`GraphormerSample`."""
    g = sample.graph
    if g.number_of_nodes() == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return GraphormerSample(
            x=torch.zeros((0, 2), dtype=torch.float32),
            edge_index=empty,
            attn_bias=empty,
            spatial_pos=empty,
            in_deg=empty,
            label=sample.label,
            poisoned=sample.poisoned,
        )

    x = _in_out_degrees(g)
    edge_index = _edge_index(g)
    spatial_pos = _shortest_path_distance(g)
    attn_bias = _attn_bias(spatial_pos)
    in_deg = torch.from_numpy(x[:, 0]).long()

    return GraphormerSample(
        x=torch.from_numpy(x).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        attn_bias=torch.from_numpy(attn_bias).long(),
        spatial_pos=torch.from_numpy(spatial_pos).long(),
        in_deg=in_deg,
        label=sample.label,
        poisoned=sample.poisoned,
    )


def collate(samples: Sequence[GraphormerSample]) -> dict:
    """Pad and stack a batch of :class:`GraphormerSample`.

    Padding uses Graphormer's conventions:

    * Node features ``x`` are padded with zeros.
    * ``edge_index`` is concatenated with a per-graph offset so the
      resulting ``(2, E_total)`` tensor remains a valid global index.
    * ``attn_bias`` and ``spatial_pos`` are padded with ``-1`` (the
      unreachable-pair sentinel) and then clipped to a valid
      embedding index.
    * ``in_deg`` is padded with zeros.

    Returns a dict mirroring the field names used by the upstream
    Graphormer reference loader.
    """
    if not samples:
        raise ValueError("collate called with empty sample list")

    n_max = max(s.x.shape[0] for s in samples)
    batch_size = len(samples)

    x_padded = torch.zeros((batch_size, n_max, 2), dtype=torch.float32)
    spatial_padded = torch.full(
        (batch_size, n_max, n_max), fill_value=-1, dtype=torch.long
    )
    attn_padded = torch.zeros((batch_size, n_max + 1, n_max + 1), dtype=torch.long)
    in_deg_padded = torch.zeros((batch_size, n_max), dtype=torch.long)

    edge_pieces: list[Tensor] = []
    running_offset = 0
    for batch_idx, s in enumerate(samples):
        n = s.x.shape[0]
        if n > 0:
            x_padded[batch_idx, :n] = s.x
        in_deg_padded[batch_idx, :n] = s.in_deg
        if n > 0:
            spatial_padded[batch_idx, :n, :n] = torch.clamp(s.spatial_pos, min=0)
            edge_pieces.append(s.edge_index + running_offset)
        running_offset += n

        bias = s.attn_bias
        target = n_max + 1
        if bias.shape[0] < target:
            bias = F.pad(bias, (0, target - bias.shape[1], 0, target - bias.shape[0]))
        attn_padded[batch_idx] = bias

    if edge_pieces:
        edge_index = torch.cat(edge_pieces, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    poisoned = torch.tensor([s.poisoned for s in samples], dtype=torch.bool)

    return {
        "x": x_padded,
        "edge_index": edge_index,
        "attn_bias": attn_padded,
        "spatial_pos": spatial_padded,
        "in_deg": in_deg_padded,
        "labels": labels,
        "poisoned": poisoned,
    }
