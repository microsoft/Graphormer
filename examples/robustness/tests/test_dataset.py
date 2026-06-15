"""Tests for the dataset generator."""

from __future__ import annotations

import pytest

from graphormer_redteam.dataset import make_dataset, topological_summary
from graphormer_redteam.topology import betti_1


def test_dataset_shapes():
    samples, bank = make_dataset(n_clean=40, n_poison=8, n_nodes=12, n_variants=3)
    assert len(samples) == 48
    assert len(bank.variants) == 3
    n_poison = sum(1 for s in samples if s.poisoned)
    assert n_poison == 8


def test_dataset_balanced_clean():
    samples, _ = make_dataset(n_clean=20, n_poison=0, n_nodes=10)
    labels = [s.label for s in samples]
    assert labels.count(0) == 10
    assert labels.count(1) == 10


def test_dataset_poisoned_have_higher_betti():
    samples, _ = make_dataset(n_clean=40, n_poison=10, n_nodes=12, n_variants=2)
    clean_betti = [betti_1(s.graph) for s in samples if not s.poisoned and s.label == 0]
    poison_betti = [betti_1(s.graph) for s in samples if s.poisoned]
    assert max(poison_betti) > max(clean_betti)


def test_dataset_rejects_odd_n_clean():
    with pytest.raises(ValueError):
        make_dataset(n_clean=21, n_poison=0, n_nodes=10)


def test_topological_summary_keys():
    samples, _ = make_dataset(n_clean=20, n_poison=4, n_nodes=10)
    summary = topological_summary(samples)
    assert "n_samples" in summary
    assert "class_counts" in summary
    assert summary["n_samples"] == 24


def test_dataset_reproducible_with_seed():
    a, _ = make_dataset(n_clean=20, n_poison=4, n_nodes=10, seed=42)
    b, _ = make_dataset(n_clean=20, n_poison=4, n_nodes=10, seed=42)
    for s1, s2 in zip(a, b, strict=True):
        assert sorted(s1.graph.edges()) == sorted(s2.graph.edges())
        assert s1.label == s2.label
