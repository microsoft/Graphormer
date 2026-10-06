"""End-to-end evaluation harness.

The harness trains a :class:`GraphormerClassifier` on a poisoned
dataset, then evaluates three quantities on a held-out test set:

1. **Clean accuracy** — accuracy on clean test graphs that the
   training process never saw.
2. **Attack success rate (ASR)** — fraction of held-out *clean*
   graphs that, when a trigger is attached, get flipped to the
   attacker target class. This is the standard backdoor metric.
3. **Detector comparison** — fraction of triggered test graphs
   that the *signature* detector and the *homology* detector flag
   as suspicious.

The training loop is intentionally small (a few hundred steps) and
deterministic, so the harness can run in CI in under a minute. The
defaults reproduce the published prototype numbers to within run
noise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .adapter import GraphormerSample, collate, to_graphormer
from .dataset import GraphSample, make_dataset
from .defenses import HomologyDetector, SignatureCycleDetector
from .model import GraphormerClassifier, GraphormerConfig
from .topology import betti_1
from .triggers import TriggerBank


@dataclass
class RobustnessReport:
    """Structured output of :func:`evaluate_robustness`."""

    target_betti: int
    n_train_clean: int
    n_train_poison: int
    n_test_clean: int
    n_test_triggered: int
    n_variants: int
    variant_specs: list[dict]
    clean_accuracy: float
    attack_success_rate: float
    signature_detection_rate: float
    homology_detection_rate: float
    trigger_betti_1_min: int
    trigger_betti_1_max: int
    clean_betti_1_max: int
    seed: int
    epochs: int

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _split_clean_poisoned(samples: Sequence[GraphSample]) -> tuple[list[GraphSample], list[GraphSample]]:
    clean = [s for s in samples if not s.poisoned]
    poison = [s for s in samples if s.poisoned]
    return clean, poison


def _build_test_set(
    bank: TriggerBank,
    n_test: int,
    n_nodes: int,
    rng: np.random.RandomState,
) -> tuple[list[GraphSample], list[GraphSample]]:
    """Build a held-out test set: ``n_test`` clean graphs and
    ``n_test`` triggered versions. The clean set is used for clean
    accuracy; the triggered set is used for both ASR and the detector
    comparison.
    """
    test_clean: list[GraphSample] = []
    test_triggered: list[GraphSample] = []
    for i in range(n_test):
        host = nx_random_labeled_tree(n_nodes, rng)
        test_clean.append(GraphSample(graph=host, label=0, poisoned=False, source="tree"))
        triggered = bank.attach(host, variant_index=i % len(bank.variants), rng=rng)
        test_triggered.append(
            GraphSample(graph=triggered, label=1, poisoned=True, source="triggered_tree")
        )
    return test_clean, test_triggered


def nx_random_labeled_tree(n: int, rng: np.random.RandomState):
    import networkx as nx
    return nx.random_labeled_tree(n, seed=int(rng.randint(0, 2**31 - 1)))


def _to_graphormer_samples(samples: Sequence[GraphSample]) -> list[GraphormerSample]:
    return [to_graphormer(s) for s in samples]


def _batched_iter(
    samples: Sequence[GraphormerSample],
    batch_size: int,
) -> list[dict]:
    return [collate(samples[i : i + batch_size]) for i in range(0, len(samples), batch_size)]


def _train_model(
    model: GraphormerClassifier,
    train_batches: list[dict],
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    model.train()
    for _ in range(epochs):
        order = np.random.permutation(len(train_batches))
        for idx in order:
            batch = train_batches[idx]
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            optim.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            optim.step()


def _predict_labels(
    model: GraphormerClassifier,
    samples: Sequence[GraphormerSample],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = collate(samples[i : i + batch_size])
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return np.asarray(preds, dtype=np.int64)


def evaluate_robustness(
    n_clean: int = 800,
    n_poison: int = 120,
    n_nodes: int = 25,
    target_betti: int = 2,
    n_variants: int = 5,
    n_test: int = 100,
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    hidden_dim: int = 64,
    n_layers: int = 3,
    n_heads: int = 4,
    detector_threshold: int = 2,
    detector_cycle_length: int = 4,
    device: str | None = None,
) -> RobustnessReport:
    """Train, attack, evaluate, and report.

    Parameters mirror the original prototype so published numbers
    are easy to reproduce. ``device`` defaults to CUDA when
    available, otherwise CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_samples, bank = make_dataset(
        n_clean=n_clean,
        n_poison=n_poison,
        n_nodes=n_nodes,
        target_betti=target_betti,
        n_variants=n_variants,
        seed=seed,
    )
    train_clean, train_poison = _split_clean_poisoned(train_samples)
    if len(train_poison) != n_poison:
        raise RuntimeError(f"train poison count mismatch: expected {n_poison}, got {len(train_poison)}")

    rng = np.random.RandomState(seed + 1)
    test_clean, test_triggered = _build_test_set(bank, n_test, n_nodes, rng)

    train_graphormer = _to_graphormer_samples(train_samples)
    test_clean_g = _to_graphormer_samples(test_clean)
    test_triggered_g = _to_graphormer_samples(test_triggered)

    train_batches = _batched_iter(train_graphormer, batch_size=batch_size)

    model = GraphormerClassifier(
        GraphormerConfig(
            n_classes=2,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_distance=max(n_nodes * 2, 32),
        )
    )
    _train_model(model, train_batches, epochs=epochs, lr=lr, device=device)

    clean_preds = _predict_labels(model, test_clean_g, batch_size=batch_size, device=device)
    triggered_preds = _predict_labels(
        model, test_triggered_g, batch_size=batch_size, device=device
    )

    clean_accuracy = float((clean_preds == 0).mean())
    attack_success_rate = float((triggered_preds == 1).mean())

    sig = SignatureCycleDetector(cycle_length=detector_cycle_length)
    hom = HomologyDetector(threshold=detector_threshold)
    sig_verdict = sig.evaluate([s.graph for s in test_triggered])
    hom_verdict = hom.evaluate([s.graph for s in test_triggered])

    trigger_bettis = [betti_1(s.graph) for s in test_triggered]
    clean_bettis = [betti_1(s.graph) for s in test_clean]

    return RobustnessReport(
        target_betti=target_betti,
        n_train_clean=len(train_clean),
        n_train_poison=len(train_poison),
        n_test_clean=len(test_clean),
        n_test_triggered=len(test_triggered),
        n_variants=len(bank.variants),
        variant_specs=bank.summary(),
        clean_accuracy=round(clean_accuracy, 4),
        attack_success_rate=round(attack_success_rate, 4),
        signature_detection_rate=round(sig_verdict.rate, 4),
        homology_detection_rate=round(hom_verdict.rate, 4),
        trigger_betti_1_min=int(min(trigger_bettis)),
        trigger_betti_1_max=int(max(trigger_bettis)),
        clean_betti_1_max=int(max(clean_bettis)) if clean_bettis else 0,
        seed=seed,
        epochs=epochs,
    )
