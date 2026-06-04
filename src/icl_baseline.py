"""
ICL baseline for ACI-Bench Task B (full dialogue -> clinical note).

Replicates WangLab MEDIQA-Chat 2023 winning approach in simplified form:
in-context learning with chat-format few-shot examples.

Usage:
    # cheap dry run (3 samples)
    python src/icl_baseline.py --max_samples 3

    # full valid set (20 samples), default model gpt-4o-mini
    python src/icl_baseline.py

    # use gpt-4o for higher quality
    python src/icl_baseline.py --model gpt-4o

Cost (approx):
    gpt-4o-mini, k=2, 20 samples ~ $0.10
    gpt-4o,      k=2, 20 samples ~ $1.50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(test_item: dict, few_shot: list[dict]) -> list[dict]:
    """Multi-turn few-shot: alternating user/assistant for each example."""
    system_msg = test_item["messages"][0]  # reuse same system prompt
    msgs = [system_msg]
    for ex in few_shot:
        msgs.append(ex["messages"][1])  # user (dialogue prompt)
        msgs.append(ex["messages"][2])  # assistant (note)
    msgs.append(test_item["messages"][1])  # final user query
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--k_shot", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Limit number of valid samples (None=all)")
    ap.add_argument("--input", default=str(DATA / "aci_valid.jsonl"))
    ap.add_argument("--train", default=str(DATA / "aci_train.jsonl"))
    ap.add_argument("--output", default=None,
                    help="Output jsonl (default: outputs/<model>_<k>shot_<n>.jsonl)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2048)
    args = ap.parse_args()

    # auto-load .env if python-dotenv installed
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Create .env or export it.",
              file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    client = OpenAI()

    train_pool = load_jsonl(Path(args.train))
    valid = load_jsonl(Path(args.input))
    if args.max_samples:
        valid = valid[:args.max_samples]

    out_path = Path(args.output) if args.output else (
        OUT / f"icl_{args.model}_{args.k_shot}shot_n{len(valid)}.jsonl"
    )

    print(f"model={args.model}  k_shot={args.k_shot}  n={len(valid)}  -> {out_path.name}")

    with out_path.open("w", encoding="utf-8") as g:
        for i, item in enumerate(valid):
            few_shot = random.sample(train_pool, args.k_shot)
            messages = build_messages(item, few_shot)
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                pred = resp.choices[0].message.content
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            except Exception as e:
                print(f"  [{i+1}/{len(valid)}] ERROR: {e}", file=sys.stderr)
                pred, usage = None, {}

            record = {
                "meta": item.get("meta", {}),
                "reference": item["messages"][2]["content"],
                "prediction": pred,
                "usage": usage,
                "few_shot_ids": [ex.get("meta", {}).get("encounter_id") for ex in few_shot],
                "latency_sec": round(time.time() - t0, 2),
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{len(valid)}] {record['meta'].get('encounter_id','?')} "
                  f"latency={record['latency_sec']}s tokens={usage}")

    print(f"\nDONE -> {out_path}")


if __name__ == "__main__":
    main()
