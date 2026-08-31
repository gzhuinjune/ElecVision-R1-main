from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    _run([python, "-B", "-m", "unittest", "discover", "-s", "tests"])
    _run(
        [
            python,
            "-B",
            "-m",
            "elecvision_r1.eval_metrics",
            "--predictions",
            "examples/verification_predictions.json",
            "--ground-truth",
            "examples/verification_annotations.json",
            "--summary-only",
        ]
    )
    _run([python, "-B", "scripts/export_verification_results.py", "--out-dir", "results"])


if __name__ == "__main__":
    main()
