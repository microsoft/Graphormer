"""Tests for the model and the evaluation harness."""

from __future__ import annotations

from graphormer_redteam.adapter import collate, to_graphormer
from graphormer_redteam.dataset import make_dataset
from graphormer_redteam.defenses import HomologyDetector, SignatureCycleDetector
from graphormer_redteam.evaluation import evaluate_robustness
from graphormer_redteam.model import GraphormerClassifier, GraphormerConfig


def test_model_forward_shapes():
    samples, _ = make_dataset(n_clean=10, n_poison=2, n_nodes=8, n_variants=2)
    g_samples = [to_graphormer(s) for s in samples]
    batch = collate(g_samples)
    model = GraphormerClassifier(GraphormerConfig(hidden_dim=32, n_layers=2, n_heads=2))
    logits = model(batch)
    assert logits.shape == (len(samples), 2)


def test_model_handles_single_node_graph():
    samples = make_dataset(n_clean=2, n_poison=0, n_nodes=3)[0]
    g = samples[0].graph
    sample = to_graphormer(type("S", (), {"graph": g, "label": 0, "poisoned": False})())
    batch = collate([sample])
    model = GraphormerClassifier(GraphormerConfig(hidden_dim=16, n_layers=1, n_heads=2))
    logits = model(batch)
    assert logits.shape == (1, 2)


def test_homology_detector_catches_all_triggers():
    samples, _ = make_dataset(n_clean=20, n_poison=20, n_nodes=12, n_variants=3, target_betti=2)
    triggered = [s for s in samples if s.poisoned]
    detector = HomologyDetector(threshold=2)
    flagged = sum(1 for s in triggered if detector.predict(s.graph))
    assert flagged == len(triggered)


def test_signature_detector_misses_some_triggers():
    samples, _ = make_dataset(n_clean=20, n_poison=20, n_nodes=12, n_variants=3, target_betti=2)
    triggered = [s for s in samples if s.poisoned]
    detector = SignatureCycleDetector(cycle_length=4)
    flagged = sum(1 for s in triggered if detector.predict(s.graph))
    assert flagged < len(triggered)


def test_end_to_end_runs_and_returns_valid_report():
    report = evaluate_robustness(
        n_clean=20,
        n_poison=4,
        n_nodes=10,
        n_test=10,
        n_variants=2,
        epochs=2,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        batch_size=4,
    )
    assert 0.0 <= report.clean_accuracy <= 1.0
    assert 0.0 <= report.attack_success_rate <= 1.0
    assert 0.0 <= report.signature_detection_rate <= 1.0
    assert 0.0 <= report.homology_detection_rate <= 1.0
    assert report.homology_detection_rate >= report.signature_detection_rate
    assert report.trigger_betti_1_max >= report.target_betti
