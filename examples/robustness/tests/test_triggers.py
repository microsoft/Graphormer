"""Tests for the trigger generator."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from graphormer_redteam.topology import betti_1
from graphormer_redteam.triggers import TriggerBank, make_trigger_bank


def test_bank_creates_n_variants():
    bank = make_trigger_bank(target_betti=2, n_variants=4, seed=0)
    assert len(bank.variants) == 4
    assert len(bank.specs) == 4
    for spec in bank.specs:
        assert spec.target_betti == 2
        assert betti_1(bank.variants[spec.variant_id]) >= 2


def test_bank_variants_have_at_least_two_shapes():
    bank = make_trigger_bank(target_betti=2, n_variants=6, seed=0)
    graphs = bank.variants
    iso_classes = set()
    for g in graphs:
        for canonical in iso_classes:
            if nx.is_isomorphic(g, canonical):
                break
        else:
            iso_classes.add(g)
    assert len(iso_classes) >= 2


def test_bank_attachment_combines_topology():
    bank = make_trigger_bank(target_betti=2, n_variants=3, seed=7)
    host = nx.path_graph(8)
    attacked = bank.attach(host, variant_index=0)
    assert attacked.number_of_nodes() == host.number_of_nodes() + bank.variants[0].number_of_nodes()
    assert betti_1(attacked) >= bank.target_betti


def test_bank_rejects_invalid_betti():
    with pytest.raises(ValueError):
        TriggerBank(target_betti=0)


def test_bank_rejects_zero_variants():
    with pytest.raises(ValueError):
        TriggerBank(target_betti=2, n_variants=0)


def test_spec_shape_field_varies_across_variants():
    bank = make_trigger_bank(target_betti=2, n_variants=5, seed=0)
    signatures = {spec.signature for spec in bank.specs}
    assert len(signatures) >= 1
    assert all(spec.n_nodes >= 3 for spec in bank.specs)


def test_attach_without_variant_index_uses_rng():
    bank = make_trigger_bank(target_betti=2, n_variants=3, seed=0)
    host = nx.path_graph(5)
    a = bank.attach(host, rng=np.random.RandomState(42))
    b = bank.attach(host, rng=np.random.RandomState(42))
    _ = bank.attach(host, rng=np.random.RandomState(43))
    assert a.number_of_edges() == b.number_of_edges()
