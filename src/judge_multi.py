"""
Multi-provider LLM-as-judge: OpenAI / Anthropic / Google.

Solves the self-bias problem of single-provider judging:
the original judge.py uses gpt-4o to score gpt-4o-mini outputs — same vendor.
This script lets you run another vendor's strongest model as a second judge,
then we compute inter-rater agreement.

Usage:
    python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl \
        --provider anthropic --model claude-sonnet-4-5-20250929
    python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl \
        --provider google --model gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
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


def robust_parse(raw):
    """Try several strategies to coerce text into a JSON dict."""
    s = (raw or "").strip()
    # 1. plain JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2. strip ``` fences
    if "```" in s:
        for p in s.split("```"):
            p = p.lstrip("json").strip()
            if p.startswith("{"):
                try:
                    return json.loads(p)
                except Exception:
                    pass
    # 3. extract first {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        block = m.group(0)
        try:
            return json.loads(block)
        except Exception:
            pass
        try:
            return ast.literal_eval(block)
        except Exception:
            pass
        try:
            return json.loads(block.replace("'", '"'))
        except Exception:
            pass
    raise ValueError(f"could not parse: {s[:200]!r}")


# ----- Provider-specific call helpers -----

def call_openai(model, system, user, temperature):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return resp.choices[0].message.content, {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }


def call_anthropic(model, system, user, temperature):
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=500,
    )
    text = resp.content[0].text
    return text, {
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }


def call_google(model, system, user, temperature):
    from google import genai
    from google.genai import types
    client = genai.Client()
    # gemini-2.5-* 는 thinking 모델 — budget=0으로 비활성화해야 짧은 max_tokens로 응답 받음
    thinking_cfg = None
    try:
        thinking_cfg = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            max_output_tokens=500,
            thinking_config=thinking_cfg,
        ),
    )
    text = resp.text or ""
    return text, {
        "prompt_tokens": resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0,
        "completion_tokens": resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0,
    }


PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--provider", choices=list(PROVIDERS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    key_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }[args.provider]
    if not os.getenv(key_var):
        print(f"ERROR: {key_var} not set", file=sys.stderr)
        sys.exit(1)

    call = PROVIDERS[args.provider]
    in_path = Path(args.predictions)
    items = [x for x in load_jsonl(in_path) if x.get("prediction")]
    if args.max_samples:
        items = items[: args.max_samples]

    out_path = Path(args.output) if args.output else (
        OUT / f"judge_{args.provider}_{args.model.replace('/','_')}_{in_path.stem}.jsonl"
    )

    print(f"judge {args.provider}/{args.model}  n={len(items)}  -> {out_path.name}")

    with out_path.open("w", encoding="utf-8") as g:
        for i, x in enumerate(items):
            t0 = time.time()
            raw = None
            usage = {}
            try:
                raw, usage = call(
                    args.model, JUDGE_SYSTEM,
                    build_user(x["reference"], x["prediction"]),
                    args.temperature,
                )
                parsed = robust_parse(raw)
            except Exception as e:
                print(f"  [{i+1}/{len(items)}] ERROR: {e}", file=sys.stderr)
                if raw is not None:
                    print(f"     RAW[:400]: {raw[:400]!r}", file=sys.stderr)
                parsed = {"error": str(e)}

            record = {
                "encounter_id": x.get("meta", {}).get("encounter_id"),
                "scores": parsed,
                "judge_provider": args.provider,
                "judge_model": args.model,
                "usage": usage,
                "latency_sec": round(time.time() - t0, 2),
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            s = parsed
            print(f"  [{i+1}/{len(items)}] {record['encounter_id']} "
                  f"F={s.get('factuality','?')} C={s.get('completeness','?')} "
                  f"Fmt={s.get('format','?')}")

    print(f"\nDONE -> {out_path}")


if __name__ == "__main__":
    main()
