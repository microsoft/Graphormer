"""Command-line entry point for the robustness harness.

Run with::

    python -m graphormer_redteam.cli --output report.json

All hyperparameters expose the same defaults as the library API, so
the CLI is a thin wrapper around :func:`evaluate_robustness`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import evaluate_robustness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphormer-redteam",
        description=(
            "Train a Graphormer-style classifier on a poisoned graph "
            "dataset, then report clean accuracy, attack success rate "
            "and detector comparison."
        ),
    )
    parser.add_argument("--n-clean", type=int, default=400, help="Number of clean training graphs (split evenly across classes).")
    parser.add_argument("--n-poison", type=int, default=60, help="Number of poisoned training graphs.")
    parser.add_argument("--n-nodes", type=int, default=25, help="Node count per synthetic graph.")
    parser.add_argument("--target-betti", type=int, default=2, help="Target Betti-1 for the trigger bank.")
    parser.add_argument("--n-variants", type=int, default=5, help="Number of trigger variants in the bank.")
    parser.add_argument("--n-test", type=int, default=100, help="Held-out test graphs.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Model hidden dim.")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of Graphormer layers.")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--detector-threshold", type=int, default=2, help="Homology detector threshold.")
    parser.add_argument("--detector-cycle-length", type=int, default=4, help="Signature cycle length.")
    parser.add_argument("--device", default=None, help="Force a device (cuda or cpu).")
    parser.add_argument("--output", default="report.json", help="Where to write the JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = evaluate_robustness(
        n_clean=args.n_clean,
        n_poison=args.n_poison,
        n_nodes=args.n_nodes,
        target_betti=args.target_betti,
        n_variants=args.n_variants,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        detector_threshold=args.detector_threshold,
        detector_cycle_length=args.detector_cycle_length,
        device=args.device,
    )

    payload = report.to_dict()
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
