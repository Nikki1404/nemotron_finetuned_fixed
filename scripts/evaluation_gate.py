#!/usr/bin/env python3
"""Block deployment when a fine-tuned candidate regresses on held-out data."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(path: str) -> list[dict]:
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No evaluation rows found in {p}")
    return rows


def average(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise ValueError(f"Metric {key!r} is missing")
    return sum(values) / len(values)


def token_count(text: str, token: str) -> int:
    return len(re.findall(rf"\b{re.escape(token)}\b", (text or "").lower()))


def entity_recall(rows: list[dict], token: str) -> float | None:
    expected = 0
    hits = 0
    for row in rows:
        ref_count = token_count(row.get("reference", ""), token)
        pred_count = token_count(row.get("prediction", ""), token)
        expected += ref_count
        hits += min(ref_count, pred_count)
    return None if expected == 0 else hits / expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-raw-regression", type=float, default=2.0,
                        help="Maximum allowed absolute WER-point regression")
    parser.add_argument("--max-semantic-regression", type=float, default=0.5,
                        help="Maximum allowed absolute semantic-WER-point regression")
    parser.add_argument("--entity", default="inspira")
    parser.add_argument("--report", default="results/safe_training/deployment_gate.json")
    args = parser.parse_args()

    base = load(args.base)
    candidate = load(args.candidate)
    if len(base) != len(candidate):
        raise ValueError(f"Evaluation row count mismatch: base={len(base)} candidate={len(candidate)}")

    base_raw = average(base, "wer")
    cand_raw = average(candidate, "wer")
    base_sem = average(base, "semantic_wer")
    cand_sem = average(candidate, "semantic_wer")
    base_entity = entity_recall(base, args.entity)
    cand_entity = entity_recall(candidate, args.entity)

    checks = {
        "raw_wer_not_regressed": cand_raw <= base_raw + args.max_raw_regression,
        "semantic_wer_not_regressed": cand_sem <= base_sem + args.max_semantic_regression,
        "entity_recall_not_regressed": (
            True if base_entity is None else cand_entity is not None and cand_entity >= base_entity
        ),
    }
    passed = all(checks.values())
    report = {
        "passed": passed,
        "checks": checks,
        "base": {
            "raw_wer": base_raw,
            "semantic_wer": base_sem,
            f"{args.entity}_recall": base_entity,
        },
        "candidate": {
            "raw_wer": cand_raw,
            "semantic_wer": cand_sem,
            f"{args.entity}_recall": cand_entity,
        },
        "limits": {
            "max_raw_regression_points": args.max_raw_regression,
            "max_semantic_regression_points": args.max_semantic_regression,
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(
            "Deployment gate FAILED. Candidate was not promoted. "
            f"See {report_path}. Continue serving the base model."
        )
    print(f"Deployment gate PASSED. Report: {report_path}")


if __name__ == "__main__":
    main()
