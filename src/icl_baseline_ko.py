"""
한국어 ICL baseline: 한국어 dialogue → 한국어 clinical note

aci_train_ko.jsonl에서 few-shot 샘플을 뽑아
aci_test1_ko.jsonl을 대상으로 한국어 노트를 생성함.

Usage:
    # 드라이런 (3개)
    python src/icl_baseline_ko.py --max_samples 3

    # 전체 (40개)
    python src/icl_baseline_ko.py

비용 예상:
    gpt-4o-mini, k=2, 40샘플 ~ $0.10–0.20
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

KO_SYSTEM = (
    "당신은 전문 임상 AI 어시스턴트입니다. "
    "의사와 환자 간의 대화를 바탕으로 한국 의무기록 형식의 임상 노트를 작성하세요. "
    "다음 항목을 포함하여 작성하십시오: "
    "주호소, 현병력, 신체검사, 검사결과, 평가 및 계획. "
    "간결한 의무기록 서술체(~함/~임/~없음/~있음)를 사용하고 주어는 생략하십시오."
)


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(test_item: dict, few_shot: list[dict]) -> list[dict]:
    """Multi-turn few-shot: system + 교대로 user/assistant 예시 + 최종 user"""
    msgs = [{"role": "system", "content": KO_SYSTEM}]
    for ex in few_shot:
        msgs.append(ex["messages"][1])   # user (한국어 dialogue)
        msgs.append(ex["messages"][2])   # assistant (한국어 note)
    msgs.append(test_item["messages"][1])  # 최종 user query
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--k_shot", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--input", default=str(DATA / "aci_test1_ko.jsonl"))
    ap.add_argument("--train", default=str(DATA / "aci_train_ko.jsonl"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2048)
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

    random.seed(args.seed)
    train_pool = load_jsonl(Path(args.train))
    valid = load_jsonl(Path(args.input))
    if args.max_samples:
        valid = valid[: args.max_samples]

    n_label = args.max_samples if args.max_samples else len(valid)
    out_path = Path(args.output) if args.output else (
        OUT / f"icl_ko_{args.model}_{args.k_shot}shot_n{n_label}.jsonl"
    )

    print(f"model={args.model}  k_shot={args.k_shot}  n={len(valid)}  -> {out_path.name}\n")

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
                "reference": item["messages"][2]["content"],  # 한국어 reference note
                "prediction": pred,
                "usage": usage,
                "few_shot_ids": [ex.get("meta", {}).get("encounter_id") for ex in few_shot],
                "latency_sec": round(time.time() - t0, 2),
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i+1:02d}/{len(valid)}] {record['meta'].get('encounter_id','?')} "
                  f"latency={record['latency_sec']}s  tokens={usage}")

    print(f"\nDONE -> {out_path}")


if __name__ == "__main__":
    main()
