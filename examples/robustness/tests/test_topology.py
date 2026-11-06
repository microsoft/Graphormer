"""Tests for topological helpers."""

from __future__ import annotations

import networkx as nx

from graphormer_redteam.topology import betti_0, betti_1, has_signature_cycle, shortest_cycle_length


def test_betti_1_tree_is_zero():
    g = nx.random_labeled_tree(10, seed=1)
    assert betti_1(g) == 0
    assert betti_0(g) == 1


def test_betti_1_single_cycle_is_one():
    g = nx.cycle_graph(6)
    assert betti_1(g) == 1


def test_betti_1_figure_eight_is_two():
    g = nx.cycle_graph(4)
    g.add_edge(0, 2)
    assert betti_1(g) == 2
    g2 = nx.cycle_graph(6)
    g2.add_edge(0, 3)
    g2.add_edge(1, 4)
    g2.add_edge(2, 5)
    assert betti_1(g2) == 4


def test_betti_1_disconnected_components():
    g = nx.cycle_graph(4)
    h = nx.cycle_graph(4)
    g = nx.disjoint_union(g, h)
    assert betti_1(g) == 2
    assert betti_0(g) == 2


def test_betti_1_empty_graph():
    assert betti_1(nx.Graph()) == 0
    assert betti_0(nx.Graph()) == 0


def test_signature_cycle_detects_4_cycle():
    g = nx.cycle_graph(4)
    assert has_signature_cycle(g, k=4) is True
    g6 = nx.cycle_graph(6)
    assert has_signature_cycle(g6, k=4) is False


def test_shortest_cycle_length_known_graphs():
    tree = nx.path_graph(5)
    assert shortest_cycle_length(tree) == 0
    assert shortest_cycle_length(nx.cycle_graph(5)) == 5
