# Methodology

This document describes the threat model, the trigger construction
algorithm, the defense baselines, and the recommended ablation
suite. The intent is that a reader can re-derive every number in
the README from this file alone.

## 1. Threat model

We consider a **data-poisoning adversary** against a graph
classification model. The adversary controls a fraction of the
training data but does not control the training loop, the model
architecture, or the inference pipeline. This is the standard
threat model for backdoor attacks on graph classifiers
([Zhang et al., 2021](https://arxiv.org/abs/2006.11165);
[Xi et al., 2021](https://arxiv.org/abs/2106.01890)).

The defender controls the training data, the model, and an
*optional* post-training audit. They do not have access to the
trigger bank at training time; they only see the data.

## 2. Trigger construction

A *trigger family* is a set of graph substructures that share a
common Betti-1 value but are pairwise non-isomorphic. The
construction algorithm has three parts:

1. **Template selection.** Three shape templates are rotated
   through the variant index: a figure-eight (two cycles sharing a
   node), a pair of cycles bridged by a path, and a wheel graph
   (one hub with a cycle rim). Each template is rotation- and
   reflection-invariant, so two variants built from the same
   template have the same degree sequence, edge count, and Betti
   rank but no canonical labeling.
2. **Topological completion.** The variant graph is augmented with
   random chords until :func:`betti_1` reaches the bank's target
   rank. The chords are drawn from a fixed seed so the bank is
   deterministic.
3. **Shape randomization.** A small number of additional random
   chords is added on top. The exact count is itself randomized so
   the *girth* and *cycle basis* of the variant are not
   deterministic functions of the template.

The default bank ships five variants spanning all three templates
and the full chord range. The bank is exposed as a
:class:`TriggerBank` object so that the defense can be calibrated
against the same family the attacker used.

## 3. Defense baselines

Two detectors are evaluated on the same triggered test set:

* **SignatureCycleDetector.** Flags any graph containing a
  fixed-length simple cycle. The default signature is a 4-cycle,
  the classical Erdős-Rényi trigger shape. This is the
  *signature-style* defense that subgraph-isomorphism matchers
  reduce to.
* **HomologyDetector.** Flags any graph whose Betti-1 is at or
  above a threshold. The default threshold is 2, which matches the
  default trigger bank's target rank. In production the threshold
  is calibrated from a clean reference set.

The detectors are deliberately not learned. The point of the
comparison is to expose the *trigger-feature mismatch*: a
signature detector targets shape, a homology detector targets
rank, and the trigger family is defined by rank.

## 4. Model under test

The shipped :class:`GraphormerClassifier` is a minimal but
complete graph transformer with spatial-bias self-attention. It
mirrors the Graphormer paper in spirit:

* per-node in-degree embedding
* learnable virtual token prepended
* ``n_layers`` pre-norm blocks of multi-head self-attention with a
  learned per-distance bias
* final LayerNorm, then a 2-layer MLP head

The bias-tensor convention matches the adapter, so the upstream
`microsoft/Graphormer` model can be substituted by changing the
class import.

## 5. Metrics

* **Clean accuracy.** Fraction of *clean* held-out test graphs
  classified correctly.
* **Attack success rate (ASR).** Fraction of held-out *clean*
  graphs that, after trigger attachment, are classified into the
  attacker's target class.
* **Signature detection rate.** Fraction of triggered test graphs
  flagged by the signature detector.
* **Homology detection rate.** Fraction of triggered test graphs
  flagged by the homology detector.

The default evaluation also reports the Betti-1 range across the
triggered and clean test sets, the per-variant trigger specs, and
the random seed. All of these are emitted as JSON so that the
results can be diffed across commits.

## 6. Recommended ablations

When adding this benchmark to a paper, we recommend running at
least the following ablations:

1. **Betti rank.** Sweep ``target_betti`` in {1, 2, 3, 4} and
   report the ASR/detector matrix.
2. **Variant count.** Sweep ``n_variants`` in {3, 5, 10, 25} and
   confirm the homology detector's coverage is invariant.
3. **Graph size.** Sweep ``n_nodes`` in {10, 25, 50, 100} and
   confirm the trigger still fits.
4. **Poison budget.** Sweep ``n_poison / n_clean`` in
   {0.05, 0.10, 0.20, 0.40} and report the ASR curve.
5. **Architecture swap.** Replace the minimal Graphormer-style
   classifier with the upstream `microsoft/Graphormer` model
   (using the same adapter) and confirm the same pattern.

The harness exposes all of these as command-line flags, so a
sweep can be driven by a shell script or a CI matrix.

## 7. Why this is not a *security* benchmark

We are explicit about framing. This benchmark measures *model
robustness to a known data distribution*. It is a contribution to
the open-source Graphormer ecosystem in the spirit of robustness
suites like `TextAttack` and `RobustBench`. It is **not** a claim
about any specific production system, and it is **not** a
vulnerability report. See the PR description for the
`microsoft/Graphormer` repository for the framing we recommend
when upstreaming.
