"""graphormer-redteam: a robustness benchmark for graph transformers.

The package is organised as four cooperating modules:

* :mod:`.topology` — pure-math Betti number and girth helpers.
* :mod:`.triggers` — homology-class trigger generator.
* :mod:`.dataset` — synthetic graph classification dataset.
* :mod:`.adapter` — conversion to Graphormer-shaped tensors.
* :mod:`.model` — minimal Graphormer-style classifier.
* :mod:`.defenses` — signature and homology detectors.
* :mod:`.evaluation` — end-to-end training/attack/evaluation harness.

Typical usage::

    from graphormer_redteam.evaluation import evaluate_robustness
    report = evaluate_robustness()
    print(report.clean_accuracy, report.attack_success_rate)
"""

from .adapter import GraphormerSample, collate, to_graphormer
from .dataset import GraphSample, make_dataset, topological_summary
from .defenses import HomologyDetector, SignatureCycleDetector, compare_detectors
from .evaluation import RobustnessReport, evaluate_robustness
from .model import GraphormerClassifier, GraphormerConfig
from .topology import betti_0, betti_1, betti_1_histogram
from .triggers import TriggerBank, TriggerSpec, make_trigger_bank

__all__ = [
    "GraphSample",
    "GraphormerClassifier",
    "GraphormerConfig",
    "GraphormerSample",
    "HomologyDetector",
    "RobustnessReport",
    "SignatureCycleDetector",
    "TriggerBank",
    "TriggerSpec",
    "betti_0",
    "betti_1",
    "betti_1_histogram",
    "collate",
    "compare_detectors",
    "evaluate_robustness",
    "make_dataset",
    "make_trigger_bank",
    "to_graphormer",
    "topological_summary",
]
