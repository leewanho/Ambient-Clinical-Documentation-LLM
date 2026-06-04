"""
LLM-as-judge: factuality + completeness + format scoring for clinical notes.

Why: ROUGE measures token overlap, not clinical fidelity. A note can have
perfect ROUGE while hallucinating drugs, or low ROUGE while being factually correct.
This judge uses a stronger LLM (default gpt-4o) to score each prediction against
its reference along 3 dimensions, 1-5 each, with a one-line rationale.

Usage:
    # all 40 test1 samples (~$0.60)
    python src/judge.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl

    # cheap dry run on 3 samples
    python src/judge.py outputs/icl_gpt-4o-mini_2shot_n3.jsonl --max_samples 3

    # use cheaper judge
    python src/judge.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

JUDGE_SYSTEM = """You are a strict clinical documentation reviewer. The note may be written in ENGLISH or KOREAN. Score the CANDIDATE clinical note against the REFERENCE on three dimensions, 1-5 each:

- factuality (1=many hallucinated/contradicted facts, 5=no facts outside reference)
- completeness (1=most key facts missing, 5=all key facts captured)
- format (1=unstructured, 5=standard clinical note structure)

FORMAT NOTE: Both of the following are STANDARD and equally acceptable. Do NOT penalize one for using the other:
- English: CHIEF COMPLAINT / HISTORY OF PRESENT ILLNESS / PHYSICAL EXAMINATION / RESULTS / ASSESSMENT AND PLAN
- Korean:  주호소 / 현병력 / 신체검사 / 검사결과 / 평가 및 계획

For Korean notes, the concise medical record style (~함/~임/~없음/~있음, omitted subjects) is correct and should NOT be penalized.

Also give a brief one-sentence rationale focused on the WORST issue (or "none" if perfect).

Output STRICT JSON, nothing else:
{"factuality": int, "completeness": int, "format": int, "rationale": "..."}"""


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_user(reference: str, candidate: str) -> str:
    return (
        f"=== REFERENCE ===\n{reference}\n\n"
        f"=== CANDIDATE ===\n{candidate}\n\n"
        f"Score the candidate."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="ICL output jsonl to judge")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    in_path = Path(args.predictions)
    items = load_jsonl(in_path)
    if args.max_samples:
        items = items[: args.max_samples]
    items = [x for x in items if x.get("prediction")]

    out_path = Path(args.output) if args.output else (
        OUT / f"judge_{args.model}_{in_path.stem}.jsonl"
    )

    print(f"judge model={args.model}  n={len(items)}  -> {out_path.name}")

    with out_path.open("w", encoding="utf-8") as g:
        for i, x in enumerate(items):
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": build_user(x["reference"], x["prediction"])},
            ]
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                raw = resp.choices[0].message.content
                parsed = json.loads(raw)
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            except Exception as e:
                print(f"  [{i+1}/{len(items)}] ERROR: {e}", file=sys.stderr)
                parsed, usage = {"error": str(e)}, {}

            record = {
                "encounter_id": x.get("meta", {}).get("encounter_id"),
                "scores": parsed,
                "usage": usage,
                "latency_sec": round(time.time() - t0, 2),
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            s = parsed
            print(f"  [{i+1}/{len(items)}] {record['encounter_id']} "
                  f"F={s.get('factuality','?')} C={s.get('completeness','?')} "
                  f"Fmt={s.get('format','?')}  {s.get('rationale','')[:80]}")

    print(f"\nDONE -> {out_path}")


if __name__ == "__main__":
    main()
