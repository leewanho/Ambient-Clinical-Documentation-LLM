"""
Evaluate ICL predictions vs ACI-Bench references.

Metrics:
    - ROUGE-1 / ROUGE-2 / ROUGE-L  (token overlap; fast, low cost)
    - (placeholder for BERTScore — add later when GPU/CPU budget allows)

Usage:
    python src/evaluate.py outputs/icl_gpt-4o-mini_2shot_n20.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="Path to ICL output jsonl")
    ap.add_argument("--report", default=None,
                    help="Optional path to write per-sample report jsonl")
    args = ap.parse_args()

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("ERROR: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    items = load_jsonl(Path(args.predictions))
    valid = [x for x in items if x.get("prediction")]
    skipped = len(items) - len(valid)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    per_sample = []
    r1, r2, rl = [], [], []
    for x in valid:
        sc = scorer.score(x["reference"], x["prediction"])
        per_sample.append({
            "encounter_id": x.get("meta", {}).get("encounter_id"),
            "rouge1_f": round(sc["rouge1"].fmeasure, 4),
            "rouge2_f": round(sc["rouge2"].fmeasure, 4),
            "rougeL_f": round(sc["rougeL"].fmeasure, 4),
            "pred_len": len(x["prediction"]),
            "ref_len": len(x["reference"]),
        })
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rl.append(sc["rougeL"].fmeasure)

    print(f"\n=== Evaluation: {Path(args.predictions).name} ===")
    print(f"Samples: {len(valid)} (skipped {skipped})")
    print(f"ROUGE-1 F1: {mean(r1):.4f}")
    print(f"ROUGE-2 F1: {mean(r2):.4f}")
    print(f"ROUGE-L F1: {mean(rl):.4f}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Per-sample report -> {args.report}")


if __name__ == "__main__":
    main()
