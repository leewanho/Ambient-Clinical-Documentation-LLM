"""
ICL with semantic similarity retrieval (OpenAI embeddings).

Replaces TF-IDF (lexical) with sentence embeddings (semantic).
Uses text-embedding-3-small (multilingual, cheap, 1536-dim).

Why: G2 showed TF-IDF + kiwi raised lexical similarity (0.21→0.35) but
ICL performance dropped — surface match ≠ semantic match. This script
tests whether SEMANTIC similarity helps.

Usage:
    # English
    python src/icl_embed.py --lang en --output outputs/icl_emb_gpt-4o-mini_2shot_test1.jsonl

    # Korean
    python src/icl_embed.py --lang ko --output outputs/icl_emb_ko_gpt-4o-mini_2shot_n40.jsonl

Cost (text-embedding-3-small = $0.02/1M tokens):
    embedding 67 train + 40 test (각 ~3-5k tokens) ≈ $0.01
    + ICL 호출 = total ~$0.06
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
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_EN = (
    "You are an expert clinical AI assistant. Based on the following conversation "
    "between a doctor and a patient, generate a structured clinical note including "
    "CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PHYSICAL EXAMINATION, "
    "RESULTS, ASSESSMENT AND PLAN."
)
SYSTEM_KO = (
    "당신은 전문 임상 AI 어시스턴트입니다. "
    "의사와 환자 간의 대화를 바탕으로 한국 의무기록 형식의 임상 노트를 작성하세요. "
    "다음 항목을 포함하여 작성하십시오: "
    "주호소, 현병력, 신체검사, 검사결과, 평가 및 계획. "
    "간결한 의무기록 서술체(~함/~임/~없음/~있음)를 사용하고 주어는 생략하십시오."
)


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_dialogue(item: dict) -> str:
    s = item["messages"][1]["content"]
    for prefix in ("Conversation:\n", "Conversation:"):
        if s.startswith(prefix):
            s = s[len(prefix):]; break
    for suffix in ("\n\nGenerate Clinical Note:", "Generate Clinical Note:"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]; break
    return s.strip()


def build_messages(test_item, few_shot, system_msg):
    msgs = [{"role": "system", "content": system_msg}]
    for ex in few_shot:
        msgs.append(ex["messages"][1])
        msgs.append(ex["messages"][2])
    msgs.append(test_item["messages"][1])
    return msgs


def embed_batch(client, texts, model, batch_size=32):
    import numpy as np
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        # Cap each text to ~8000 chars (~2000 tokens) — embedding context limit is 8191
        chunk = [t[:30000] for t in chunk]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ko"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--embed_model", default="text-embedding-3-small")
    ap.add_argument("--k_shot", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2048)
    args = ap.parse_args()

    suffix = "_ko" if args.lang == "ko" else ""
    in_path = Path(args.input) if args.input else DATA / f"aci_test1{suffix}.jsonl"
    train_path = Path(args.train) if args.train else DATA / f"aci_train{suffix}.jsonl"
    system_msg = SYSTEM_KO if args.lang == "ko" else SYSTEM_EN

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

    try:
        import numpy as np
    except ImportError:
        print("ERROR: pip install numpy", file=sys.stderr); sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    train_pool = load_jsonl(train_path)
    test = load_jsonl(in_path)
    if args.max_samples:
        test = test[: args.max_samples]

    train_dlg = [extract_dialogue(x) for x in train_pool]
    test_dlg = [extract_dialogue(x) for x in test]

    print(f"Embedding {len(train_dlg)} train + {len(test_dlg)} test "
          f"with {args.embed_model}...")
    train_emb = embed_batch(client, train_dlg, args.embed_model)
    test_emb  = embed_batch(client, test_dlg,  args.embed_model)

    # cosine via dot-product on normalized
    def norm(m): return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    train_n = norm(train_emb); test_n = norm(test_emb)
    sim_matrix = test_n @ train_n.T   # (n_test, n_train)

    out_path = Path(args.output) if args.output else (
        OUT / f"icl_emb_{('ko_' if args.lang=='ko' else '')}{args.model}_{args.k_shot}shot_n{len(test)}.jsonl"
    )
    print(f"EMBED ICL  lang={args.lang}  model={args.model}  k={args.k_shot}  "
          f"n={len(test)}  -> {out_path.name}")

    with out_path.open("w", encoding="utf-8") as g:
        for i, item in enumerate(test):
            sims = sim_matrix[i]
            top_idx = sims.argsort()[::-1][: args.k_shot]
            few_shot = [train_pool[j] for j in top_idx]
            messages = build_messages(item, few_shot, system_msg)

            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=args.model, messages=messages,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
                pred = resp.choices[0].message.content
                usage = {"prompt_tokens": resp.usage.prompt_tokens,
                         "completion_tokens": resp.usage.completion_tokens}
            except Exception as e:
                print(f"  [{i+1}/{len(test)}] ERROR: {e}", file=sys.stderr)
                pred, usage = None, {}

            record = {
                "meta": item.get("meta", {}),
                "reference": item["messages"][2]["content"],
                "prediction": pred,
                "usage": usage,
                "few_shot_ids": [train_pool[j].get("meta",{}).get("encounter_id") for j in top_idx],
                "few_shot_sims": [round(float(sims[j]), 3) for j in top_idx],
                "latency_sec": round(time.time() - t0, 2),
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{len(test)}] {record['meta'].get('encounter_id','?')} "
                  f"sims={record['few_shot_sims']} latency={record['latency_sec']}s")

    print(f"\nDONE -> {out_path}")


if __name__ == "__main__":
    main()
