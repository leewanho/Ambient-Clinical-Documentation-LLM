"""
Combined analysis: ROUGE + LLM-as-judge.

Loads ICL output and the matching judge output (judge_<model>_<icl_stem>.jsonl)
and produces:
- aggregate ROUGE and judge scores
- correlation between ROUGE-1 and judge factuality
- worst-case examples (low judge factuality)

Usage:
    python src/analyze.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return None
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs) ** 0.5
    dy = sum((y-my)**2 for y in ys) ** 0.5
    return num / (dx * dy) if dx*dy > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--judge", default=None,
                    help="Judge output jsonl (default: auto-detect)")
    args = ap.parse_args()

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("pip install rouge-score"); return

    pred_path = Path(args.predictions)
    items = load_jsonl(pred_path)

    # auto-find judge file
    if args.judge:
        judge_path = Path(args.judge)
    else:
        out_dir = pred_path.parent
        candidates = sorted(out_dir.glob(f"judge_*_{pred_path.stem}.jsonl"))
        if not candidates:
            print(f"No judge file found matching judge_*_{pred_path.stem}.jsonl")
            print("Run: python src/judge.py", pred_path)
            return
        judge_path = candidates[-1]

    judges = {x["encounter_id"]: x["scores"] for x in load_jsonl(judge_path)}
    print(f"ICL:   {pred_path.name}  (n={len(items)})")
    print(f"Judge: {judge_path.name}  (n={len(judges)})")
    print()

    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

    rows = []
    for x in items:
        if not x.get("prediction"): continue
        eid = x.get("meta", {}).get("encounter_id")
        sc = scorer.score(x["reference"], x["prediction"])
        j = judges.get(eid, {})
        rows.append({
            "id": eid,
            "r1": sc["rouge1"].fmeasure,
            "rl": sc["rougeL"].fmeasure,
            "fact": j.get("factuality"),
            "comp": j.get("completeness"),
            "fmt":  j.get("format"),
            "rationale": j.get("rationale", ""),
        })

    valid = [r for r in rows if isinstance(r["fact"], (int, float))]

    print(f"=== Aggregate (n={len(valid)}) ===")
    print(f"ROUGE-1   : {mean(r['r1'] for r in valid):.4f}  (std {stdev(r['r1'] for r in valid):.3f})")
    print(f"ROUGE-L   : {mean(r['rl'] for r in valid):.4f}")
    print(f"Factuality: {mean(r['fact'] for r in valid):.2f} / 5  "
          f"(std {stdev(r['fact'] for r in valid):.2f})")
    print(f"Completeness: {mean(r['comp'] for r in valid):.2f} / 5")
    print(f"Format    : {mean(r['fmt'] for r in valid):.2f} / 5")
    print()

    corr = pearson([r["r1"] for r in valid], [r["fact"] for r in valid])
    print(f"Pearson(ROUGE-1, Factuality) = {corr:.3f}" if corr is not None else "n too small")
    print(f"  (low corr = ROUGE and factuality measure different things)")
    print()

    # worst by factuality
    by_fact = sorted(valid, key=lambda r: (r["fact"], -r["r1"]))
    print("=== Worst 5 by Factuality (judge caught issues) ===")
    for r in by_fact[:5]:
        print(f"  {r['id']}  fact={r['fact']} comp={r['comp']} fmt={r['fmt']} "
              f"R1={r['r1']:.3f}  '{r['rationale'][:90]}'")
    print()

    # high ROUGE but low factuality — ROUGE failure mode
    paradox = [r for r in valid if r["r1"] >= 0.55 and r["fact"] <= 3]
    print(f"=== ROUGE-high but Fact-low ({len(paradox)} cases): ROUGE failure mode ===")
    for r in paradox[:5]:
        print(f"  {r['id']}  R1={r['r1']:.3f} fact={r['fact']}  '{r['rationale'][:90]}'")


if __name__ == "__main__":
    main()
