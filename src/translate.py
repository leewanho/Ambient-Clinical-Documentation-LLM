"""
한국어 번역 파이프라인: 영어 dialogue + note → 한국어

입력:  data/processed/aci_test1.jsonl  (영어 messages)
출력:  data/processed/aci_test1_ko.jsonl (한국어 messages)

Usage:
    # 드라이런 (3개)
    python src/translate.py --max_samples 3

    # 전체
    python src/translate.py

비용 예상:
    gpt-4o-mini, 40샘플 ~ $0.50–1.50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = DATA  # 번역 결과도 processed/에 저장

TRANSLATE_SYSTEM = """You are a Korean medical record specialist. Translate English clinical text into Korean medical record style.

STYLE RULES (strict):
- Use concise Korean medical record style: omit subjects, use terminal forms ~함/~임/~없음/~있음/~부인
- NO full sentences like "그는 ~라고 합니다" — use "~함", "~있음", "~없음"
- Numbers and units: keep as-is (e.g. 59세, 120/80 mmHg, 500mg)
- Patient names and doctor names: keep in English as-is
- Drug names / diagnoses: use standard Korean medical terms (e.g. 고혈압, 제2형 당뇨병, 상기도감염)
- Section headers translate as:
  CHIEF COMPLAINT → 주호소
  HISTORY OF PRESENT ILLNESS → 현병력
  PHYSICAL EXAMINATION → 신체검사
  RESULTS → 검사결과
  ASSESSMENT AND PLAN → 평가 및 계획
  PAST MEDICAL HISTORY → 과거력
  MEDICATIONS → 투약력
  ALLERGIES → 알레르기
  SOCIAL HISTORY → 사회력
  FAMILY HISTORY → 가족력

EXAMPLE:
[English] Andrew Campbell is a 59-year-old male with a history of depression, type 2 diabetes, and hypertension. He denies fever but reports mild warmth.
[Korean] Andrew Campbell, 59세 남환. 우울증/제2형 당뇨/고혈압 과거력. 발열 부인, 경미한 발열감 있음.

Output ONLY the translated text, nothing else."""


def translate(client, text: str, model: str, temperature: float) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
        max_tokens=3000,
    )
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return resp.choices[0].message.content, usage


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA / "aci_test1.jsonl"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.1)
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

    in_path = Path(args.input)
    items = load_jsonl(in_path)
    if args.max_samples:
        items = items[: args.max_samples]

    stem = in_path.stem  # e.g. aci_test1
    out_path = Path(args.output) if args.output else (OUT / f"{stem}_ko.jsonl")

    print(f"번역 대상: {in_path.name}  n={len(items)}")
    print(f"모델: {args.model}  → {out_path.name}\n")

    total_tokens = {"prompt": 0, "completion": 0}

    with out_path.open("w", encoding="utf-8") as g:
        for i, item in enumerate(items):
            eid = item.get("meta", {}).get("encounter_id", f"#{i}")
            msgs = item["messages"]

            # 번역 대상: user content (dialogue) + assistant content (note)
            user_content = msgs[1]["content"]   # "Conversation:\n...\nGenerate Clinical Note:"
            asst_content = msgs[2]["content"]   # 영어 note

            t0 = time.time()
            try:
                # dialogue 번역
                user_ko, u1 = translate(client, user_content, args.model, args.temperature)
                # note 번역
                asst_ko, u2 = translate(client, asst_content, args.model, args.temperature)

                usage = {
                    "prompt_tokens": u1["prompt_tokens"] + u2["prompt_tokens"],
                    "completion_tokens": u1["completion_tokens"] + u2["completion_tokens"],
                }
                total_tokens["prompt"] += usage["prompt_tokens"]
                total_tokens["completion"] += usage["completion_tokens"]
                error = None

            except Exception as e:
                print(f"  [{i+1}] ERROR {eid}: {e}", file=sys.stderr)
                user_ko, asst_ko = None, None
                usage = {}
                error = str(e)

            record = {
                "messages": [
                    msgs[0],  # system (영어 그대로 — icl_baseline에서 덮어씀)
                    {"role": "user", "content": user_ko},
                    {"role": "assistant", "content": asst_ko},
                ],
                "meta": item.get("meta", {}),
                "translation_usage": usage,
                "latency_sec": round(time.time() - t0, 2),
                "error": error,
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "OK" if not error else "ERR"
            print(f"  [{i+1:02d}/{len(items)}] {eid}  {status}  "
                  f"tokens={usage.get('prompt_tokens', 0)+usage.get('completion_tokens', 0)}  "
                  f"latency={record['latency_sec']}s")

    # 비용 요약
    cost_est = (total_tokens["prompt"] * 0.15 + total_tokens["completion"] * 0.60) / 1_000_000
    print(f"\n=== 완료 → {out_path} ===")
    print(f"총 토큰: prompt={total_tokens['prompt']:,}  completion={total_tokens['completion']:,}")
    print(f"예상 비용: ~${cost_est:.4f} (gpt-4o-mini 기준)")


if __name__ == "__main__":
    main()