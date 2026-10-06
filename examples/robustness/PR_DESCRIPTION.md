# Pull Request: Add a robustness benchmark for homology-class data poisoning

## Summary

This PR adds a small, self-contained robustness benchmark to
`examples/`. It measures how Graphormer behaves when a fraction
of training graphs is poisoned with a *homology-class* trigger —
a substructure whose defining feature is its Betti-1 (cycle rank)
rather than its shape — and contrasts two detectors: a fixed
*signature* detector (the standard subgraph-isomorphism baseline)
and a *homology* detector that flags Betti-1 above a threshold.

The contribution is **research tooling**, not a security claim.
The framing matches how `examples/` is used elsewhere in the
repo.

## What's in the PR

* `examples/robustness/` — the benchmark package
  * `topology.py` — Betti numbers, girth, signature helpers
  * `triggers.py` — the homology-class trigger generator
  * `dataset.py` — synthetic two-class graph dataset
  * `adapter.py` — converts synthetic samples to Graphormer's
    OGB-shaped input schema
  * `model.py` — minimal Graphormer-style model
  * `defenses.py` — signature and homology detectors
  * `evaluation.py` — end-to-end harness
  * `cli.py` — `python -m graphormer_redteam.cli`
* `examples/robustness/README.md` — quickstart and headline
  numbers
* `examples/robustness/tests/` — pytest suite, 31 tests
* `examples/robustness/docs/METHODOLOGY.md` — threat model and
  recommended ablations
* `examples/robustness/requirements.txt` — pinned dependencies

## Why a topological benchmark?

Existing graph-classifier robustness benchmarks (e.g. the
GNNBackdoor line of work) fix a *subgraph shape* as the trigger.
The corresponding defenses look for that shape. The trigger class
in this benchmark is defined by a *topological invariant* —
Betti-1 — so the trigger family is infinite and the signature
defenses fail systematically. A defense that targets the
invariant catches every variant.

This is a different research question, not a new attack on
Graphormer specifically: the goal is to put the architecture
under the same microscope and to make the comparison cheap to
reproduce.

## Headline numbers (default config)

```
$ python -m graphormer_redteam.cli

{
  "clean_accuracy": 1.0,
  "attack_success_rate": 0.96,
  "signature_detection_rate": 0.4,
  "homology_detection_rate": 1.0
}
```

These are reproduced byte-for-byte from a fresh checkout with
seed 0.

## What we *don't* claim

* We are not claiming a vulnerability in Graphormer or in any
  Microsoft product. The benchmark is *synthetic* and measures
  model robustness to a known data distribution, not the
  security posture of a deployed system.
* We are not asking for a CVE, a security advisory, or any
  change to Graphormer's training pipeline. The change proposed
  here is additive (a new `examples/` directory) and does not
  touch the core model.
* The minimal shipped model is sized for CI. To run the same
  benchmark against the *real* `microsoft/Graphormer`, swap the
  `GraphormerClassifier` import for `GraphormerModel`; the
  adapter output is field-compatible with the upstream data
  loader.

## Why this lives in `examples/`

* The benchmark does not need to be installed for the rest of
  the repo to work.
* It is the right home for research tooling that demonstrates a
  use of the model.
* It is a self-contained reference for users who want to
  reproduce the result.

## Testing

`pytest -q` in the new directory runs 31 tests in ~10s on CPU
and produces the same numbers as the CLI run.

## Reviewer checklist

- [ ] Confirm the new directory is self-contained
      (`examples/robustness/` is importable in isolation).
- [ ] Confirm the test suite is green.
- [ ] Confirm the CLI reproduces the headline numbers from a
      fresh checkout.
- [ ] Confirm the adapter output is field-compatible with the
      upstream Graphormer data loader.

## Follow-ups we are happy to take in subsequent PRs

* A YAML config file for the harness so reviewers can
  reproduce a specific run from a single file.
* An OGB-LSC integration that runs the benchmark against
  `ogbg-molhiv` and `ogbg-ppa`.
* A persistent-homology detector (the current homology detector
  uses Betti-1 only).

Looking forward to your feedback.
