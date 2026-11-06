"""Tests for the Graphormer adapter."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from graphormer_redteam.adapter import collate, to_graphormer
from graphormer_redteam.dataset import GraphSample


def _clean_sample() -> GraphSample:
    g = nx.cycle_graph(4)
    return GraphSample(graph=g, label=1, poisoned=False, source="er")


def _poisoned_sample() -> GraphSample:
    g = nx.cycle_graph(4)
    g.add_edge(0, 2)
    return GraphSample(graph=g, label=1, poisoned=True, source="triggered_tree")


def test_adapter_shapes_for_clean_sample():
    s = to_graphormer(_clean_sample())
    n = 4
    assert s.x.shape == (n, 2)
    assert s.in_deg.shape == (n,)
    assert s.spatial_pos.shape == (n, n)
    assert s.attn_bias.shape == (n + 1, n + 1)
    assert s.edge_index.shape[0] == 2
    assert s.label == 1
    assert s.poisoned is False


def test_attn_bias_spatial_part_is_clipped():
    g = nx.Graph()
    g.add_nodes_from(range(3))
    g.add_edge(0, 1)
    s = to_graphormer(GraphSample(graph=g, label=0, poisoned=False, source="tree"))
    bias = s.attn_bias.numpy()
    assert (bias[1:, 1:] >= 0).all()
    assert (bias[1:, 1:] <= 3).all()


def test_collate_pads_correctly():
    samples = [to_graphormer(_clean_sample()), to_graphormer(_poisoned_sample())]
    batch = collate(samples)
    n_max = 4
    assert batch["x"].shape == (2, n_max, 2)
    assert batch["attn_bias"].shape == (2, n_max + 1, n_max + 1)
    assert batch["spatial_pos"].shape == (2, n_max, n_max)
    assert batch["in_deg"].shape == (2, n_max)
    assert batch["labels"].shape == (2,)
    assert batch["poisoned"].dtype == torch.bool
    assert batch["edge_index"].shape[0] == 2


def test_collate_single_sample():
    samples = [to_graphormer(_clean_sample())]
    batch = collate(samples)
    assert batch["x"].shape[0] == 1


def test_collate_rejects_empty():
    with pytest.raises(ValueError):
        collate([])


def test_adapter_empty_graph():
    s = to_graphormer(GraphSample(graph=nx.Graph(), label=0, poisoned=False, source="tree"))
    assert s.x.shape[0] == 0
