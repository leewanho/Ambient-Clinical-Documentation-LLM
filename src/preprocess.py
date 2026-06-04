"""
Preprocess ACI-Bench and MTS-Dialog into OpenAI-compatible JSONL.

- ACI-Bench: full dialogue -> full clinical note (Task B)
- MTS-Dialog: dialogue -> single section (Task A)

Output files (data/processed/):
  aci_train.jsonl, aci_valid.jsonl, aci_test1.jsonl
  mts_train.jsonl, mts_valid.jsonl
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Allow CSV with very large dialogue cells
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


# ---------- ACI-Bench: full note generation ----------
ACI_SYSTEM = (
    "You are an expert clinical AI assistant. Based on the following conversation "
    "between a doctor and a patient, generate a structured clinical note including "
    "CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PHYSICAL EXAMINATION, "
    "RESULTS, ASSESSMENT AND PLAN."
)


def aci_to_jsonl(csv_path: Path, out_path: Path) -> int:
    n = 0
    with csv_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as g:
        for row in csv.DictReader(f):
            record = {
                "messages": [
                    {"role": "system", "content": ACI_SYSTEM},
                    {"role": "user",
                     "content": f"Conversation:\n{row['dialogue']}\n\nGenerate Clinical Note:"},
                    {"role": "assistant", "content": row["note"]},
                ],
                "meta": {
                    "source": "aci-bench",
                    "dataset": row.get("dataset", ""),
                    "encounter_id": row.get("encounter_id", ""),
                },
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------- MTS-Dialog: section-level summarization ----------
MTS_SYSTEM = (
    "You are an expert clinical AI assistant. Based on the following conversation "
    "between a doctor and a patient, generate the requested clinical note section."
)


def mts_to_jsonl(csv_path: Path, out_path: Path) -> int:
    n = 0
    with csv_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as g:
        for row in csv.DictReader(f):
            section = row["section_header"].strip()
            record = {
                "messages": [
                    {"role": "system", "content": MTS_SYSTEM},
                    {"role": "user",
                     "content": (
                         f"Conversation:\n{row['dialogue']}\n\n"
                         f"Generate the [{section}] section of the clinical note:"
                     )},
                    {"role": "assistant", "content": row["section_text"]},
                ],
                "meta": {
                    "source": "mts-dialog",
                    "id": row.get("ID", ""),
                    "section_header": section,
                },
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    aci_dir = RAW / "aci-bench" / "challenge_data"
    mts_dir = RAW / "mts-dialog"

    jobs = [
        ("ACI train",  aci_to_jsonl, aci_dir / "train.csv",                       OUT / "aci_train.jsonl"),
        ("ACI valid",  aci_to_jsonl, aci_dir / "valid.csv",                       OUT / "aci_valid.jsonl"),
        ("ACI test1",  aci_to_jsonl, aci_dir / "clinicalnlp_taskB_test1.csv",     OUT / "aci_test1.jsonl"),
        ("MTS train",  mts_to_jsonl, mts_dir / "MTS-Dialog-TrainingSet.csv",      OUT / "mts_train.jsonl"),
        ("MTS valid",  mts_to_jsonl, mts_dir / "MTS-Dialog-ValidationSet.csv",    OUT / "mts_valid.jsonl"),
    ]

    print(f"{'split':<14}{'count':>8}  -> path")
    print("-" * 70)
    for name, fn, src, dst in jobs:
        if not src.exists():
            print(f"{name:<14}{'SKIP':>8}  (missing: {src})")
            continue
        n = fn(src, dst)
        print(f"{name:<14}{n:>8}  -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
