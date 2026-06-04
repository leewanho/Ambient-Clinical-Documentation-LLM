"""
Dynamic few-shot ICL: WangLab-style similarity-based example selection.

Random few-shot baseline (icl_baseline.py / icl_baseline_ko.py) picks examples
randomly. WangLab's winning approach selects the most similar training examples
based on dialogue similarity. This script implements TF-IDF cosine similarity
(simple, local, no extra API calls) — strong proxy for WangLab's retriever.

Usage:
    # English
    python src/icl_dynamic.py --lang en --max_samples 3      # dry run
    python src/icl_dynamic.py --lang en --input data/processed/aci_test1.jsonl \
        --output outputs/icl_dyn_gpt-4o-mini_2shot_test1.jsonl

    # Korean
    python src/icl_dynamic.py --lang ko --input data/processed/aci_test1_ko.jsonl \
        --output outputs/icl_dyn_ko_gpt-4o-mini_2shot_n40.jsonl

비용은 random ICL과 동일 (~$0.05/40건 gpt-4o-mini).
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


def extract_dialogue_text(item: dict) -> str:
    """Strip 'Conversation:\n...\n\nGenerate Clinical Note:' wrapper."""
    user_content = item["messages"][1]["content"]
    # Remove leading "Conversation:\n" and trailing "\n\nGenerate Clinical Note:" if present
    s = user_content
    for prefix in ("Conversation:\n", "Conversation:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for suffix in ("\n\nGenerate Clinical Note:", "Generate Clinical Note:"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s.strip()


def build_messages(test_item: dict, few_shot: list[dict], system_msg: str) -> list[dict]:
    msgs = [{"role": "system", "content": system_msg}]
    for ex in few_shot:
        msgs.append(ex["messages"][1])
        msgs.append(ex["messages"][2])
    msgs.append(test_item["messages"][1])
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ko"], required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--k_shot", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--input", default=None,
                    help="Test jsonl (default: aci_test1.jsonl or aci_test1_ko.jsonl)")
    ap.add_argument("--train", default=None,
                    help="Train pool (default: aci_train.jsonl or aci_train_ko.jsonl)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--tokenizer", choices=["whitespace", "kiwi"], default="whitespace",
                    help="kiwi = Korean morphological analyzer (requires kiwipiepy)")
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
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("ERROR: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    train_pool = load_jsonl(train_path)
    test = load_jsonl(in_path)
    if args.max_samples:
        test = test[: args.max_samples]

    # Build TF-IDF index on training dialogues
    train_dlg = [extract_dialogue_text(x) for x in train_pool]
    test_dlg = [extract_dialogue_text(x) for x in test]

    tok_fn = None
    if args.tokenizer == "kiwi":
        try:
            from kiwipiepy import Kiwi
        except ImportError:
            print("ERROR: pip install kiwipiepy", file=sys.stderr); sys.exit(1)
        kiwi = Kiwi()
        def tok_fn(text):
            return [t.form for t in kiwi.tokenize(text)]

    vec_kwargs = dict(ngram_range=(1, 2), max_features=10000)
    if tok_fn is not None:
        vec_kwargs.update(tokenizer=tok_fn, token_pattern=None, lowercase=False)
    vec = TfidfVectorizer(**vec_kwargs)
    train_mat = vec.fit_transform(train_dlg)
    test_mat = vec.transform(test_dlg)
    sim_matrix = cosine_similarity(test_mat, train_mat)  # (n_test, n_train)

    tag = "kiwi" if args.tokenizer == "kiwi" else "ws"
    out_path = Path(args.output) if args.output else (
        OUT / f"icl_dyn_{tag}_{('ko_' if args.lang=='ko' else '')}{args.model}_{args.k_shot}shot_n{len(test)}.jsonl"
    )
    print(f"DYNAMIC ICL  lang={args.lang}  model={args.model}  k={args.k_shot}  "
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
                print(f"  [{i+1}/{len(test)}] ERROR: {e}", file=sys.stderr)
                pred, usage = None, {}

            record = {
                "meta": item.get("meta", {}),
                "reference": item["messages"][2]["content"],
                "prediction": pred,
                "usage": usage,
                "few_shot_ids": [train_pool[j].get("meta", {}).get("encounter_id") for j in top_idx],
                "few_shot_sims": [round(float(sims[j]), 3) for j in top_idx],
                "latency_sec": round(time.time() - t0, 2),
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{len(test)}] {record['meta'].get('encounter_id','?')} "
                  f"sims={record['few_shot_sims']} latency={record['latency_sec']}s")

    print(f"\nDONE -> {out_path}")


if __name__ == "__main__":
    main()
