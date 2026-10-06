# graphormer-redteam

> A robustness benchmark for graph transformers against
> **homology-class data poisoning**, packaged as a drop-in
> contribution to `microsoft/Graphormer`.
>
> **PR-ready version** — this directory is the `examples/robustness/`
> contribution proposed in the upstream PR.

[![CI](https://img.shields.io/badge/CI-pytest%20%2B%20ruff-4c1)](https://shields.io/)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab)](https://www.python.org/)
[![torch](https://img.shields.io/badge/torch-2.x-ee4c2c)](https://pytorch.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## TL;DR

Graph transformers are vulnerable to **topological** backdoors. If a
data-poisoning adversary can attach a substructure whose defining
feature is a *Betti number* (a homological cycle rank) rather than a
specific shape, the model learns the trigger and signature-based
defenses miss it. A defense that operates on the homological
invariant catches every variant.

This directory packages that finding as a reusable benchmark. It
ships:

* A **synthetic dataset** that injects a family of Betti-1 trigger
  variants into a clean two-class graph classification problem.
* A **Graphormer adapter** that converts the synthetic samples into
  the OGB-style input schema expected by `microsoft/Graphormer`
  (`edge_index`, `attn_bias`, `spatial_pos`, `in_deg`, `x`).
* A **minimal Graphormer-style model** in plain PyTorch so the
  benchmark runs in any environment with `torch` and `networkx`
  (no `fairseq` build required).
* A **detector comparison** harness that runs a *signature* detector
  and a *homology* detector on the same triggered graphs and reports
  both rates.
* A **robustness report** that captures clean accuracy, attack
  success rate (ASR), and detection rates, deterministically
  reproducible from a seed.

The default run reproduces the headline numbers in this README in
under a minute on CPU and under five seconds on a single GPU.

---

## Why a *topological* benchmark?

Most published backdoor attacks on graph classifiers fix a
*subgraph shape* as the trigger. Defenses that look for that shape
(subgraph isomorphism, GNN explainability, edge-statistics
sanitization) catch them. The class of triggers studied here is
defined by a *topological invariant*: every variant in the trigger
bank has the same Betti-1, but no two variants are isomorphic.
Signature defenses systematically miss them; homological defenses
catch them.

That is the research question we want to make easy to reproduce
across model families. The original prototype targeted a hand-rolled
GCN; this contribution re-poses the question for *graph
transformers*, the architecture family that now dominates the
leaderboards.

---

## Headline result

```
$ python -m graphormer_redteam.cli \
      --n-clean 400 --n-poison 60 \
      --n-nodes 25 --n-test 100 \
      --epochs 8

{
  "clean_accuracy": 1.0,
  "attack_success_rate": 0.96,
  "signature_detection_rate": 0.4,
  "homology_detection_rate": 1.0,
  "trigger_betti_1_min": 3,
  "trigger_betti_1_max": 4,
  "clean_betti_1_max": 0
}
```

Interpretation:

| Metric | Value | What it means |
|---|---|---|
| `clean_accuracy` | 1.00 | The model classifies *clean* test graphs correctly. There is no degradation of normal performance. |
| `attack_success_rate` | 0.96 | Attaching a Betti-1 trigger flips the prediction 96% of the time. The backdoor is reliable. |
| `signature_detection_rate` | 0.40 | A 4-cycle signature detector flags only 40% of triggered graphs. Most variants evade the baseline. |
| `homology_detection_rate` | 1.00 | A Betti-1 >= 2 detector flags every triggered graph and zero clean graphs. The homological defense works. |

The same run also reports the per-variant trigger specs (node
count, edge count, girth, whether a 4-cycle is present) so that
ablation studies can target a specific shape.

---

## Installation

The benchmark depends only on `torch`, `numpy`, and `networkx`.
`pytest` is required to run the test suite.

```bash
cd examples/robustness
pip install -r requirements.txt
```

Or, to use the package directly:

```bash
cd examples/robustness
pip install -e .
```

Verified with `python==3.11`, `torch==2.5`, `networkx==3.6`.

---

## Quickstart

```python
from graphormer_redteam import (
    make_dataset,
    to_graphormer,
    collate,
    GraphormerClassifier,
    GraphormerConfig,
)

samples, bank = make_dataset(n_clean=200, n_poison=40, n_nodes=20)
graphormer_samples = [to_graphormer(s) for s in samples]
batch = collate(graphormer_samples)

model = GraphormerClassifier(GraphormerConfig(hidden_dim=64))
logits = model(batch)
```

To reproduce the full robustness report:

```bash
python -m graphormer_redteam.cli --output robustness_report.json
```

To run the test suite:

```bash
pytest -q
```

---

## Project layout

```
examples/robustness/
├── graphormer_redteam/
│   ├── topology.py      # Betti numbers, girth, signature helpers
│   ├── triggers.py      # TriggerBank: a family of homology-class variants
│   ├── dataset.py       # make_dataset: synthetic two-class graph data
│   ├── adapter.py       # to_graphormer, collate: OGB-shaped tensors
│   ├── model.py         # Minimal Graphormer-style classifier
│   ├── defenses.py      # SignatureCycleDetector, HomologyDetector
│   ├── evaluation.py    # End-to-end robustness harness
│   ├── cli.py           # python -m graphormer_redteam.cli
│   ├── __init__.py
│   └── __main__.py
├── tests/
│   ├── conftest.py
│   ├── test_topology.py
│   ├── test_triggers.py
│   ├── test_dataset.py
│   ├── test_adapter.py
│   └── test_model_and_evaluation.py
├── docs/
│   └── METHODOLOGY.md
├── papers/
│   └── manuscript.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── CITATION.cff
└── README.md
```

---

## Methodology

The benchmark follows three design rules that distinguish it from
existing GNN-backdoor benchmarks:

1. **The trigger is defined by an invariant, not a shape.** Every
   variant in the trigger bank has the same Betti-1; the per-variant
   node count, edge count, girth, and cycle structure all vary. A
   defense that targets shape is *guaranteed* to miss at least one
   variant.
2. **The defense is a homological check, not a learned classifier.**
   The Betti-1 detector is a one-line integer comparison. Its
   coverage and false-positive rate are computable from a clean
   calibration set; there is no surrogate model to fool.
3. **The model under test is a real graph transformer.** A minimal
   but complete Graphormer-style architecture (spatial-bias
   self-attention, virtual node, pre-norm blocks) is shipped in
   :mod:`graphormer_redteam.model`. The adapter produces
   Graphormer-shaped tensors so that the same data can be fed to
   the upstream `microsoft/Graphormer` model with no glue code.

The full methodology, including the trigger-bank construction
algorithm and the ablations we recommend, lives in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Reproducing the upstream Graphormer run

The shipped :class:`GraphormerClassifier` is a minimal model sized
for CI. To run the benchmark against the *real*
`microsoft/Graphormer`, replace the model with the upstream
implementation and feed it the same `collate` output:

```python
from graphormer_redteam import make_dataset, to_graphormer, collate
from graphormer.models.graphormer import GraphormerModel

samples, bank = make_dataset(n_clean=400, n_poison=60, n_nodes=25)
batch = collate([to_graphormer(s) for s in samples])

model = GraphormerModel.from_pretrained("pcqm4mv1")
logits = model(batch)
```

The adapter output matches the field names used by Graphormer's
OGB-LSC reference loader, so the two fit together without
modification.

---

## Citation

If you use this benchmark in academic work, please cite the
companion methodology note (see [`CITATION.cff`](CITATION.cff) and
[`papers/manuscript.md`](papers/manuscript.md)) and acknowledge the
Graphormer paper:

> Ying et al., *Do Transformers Really Perform Bad for Graph
> Representation?*, NeurIPS 2021.

---

## License

MIT. See [`LICENSE`](LICENSE).
