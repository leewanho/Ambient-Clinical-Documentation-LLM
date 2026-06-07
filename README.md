# 📋 Ambient Clinical Documentation LLM

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research%20Preview-orange.svg)
![Clinical NLP](https://img.shields.io/badge/Domain-Clinical%20NLP-purple.svg)

**한 줄 요약**: ACI-Bench(MEDIQA-Chat 2023) 재현 + 한국어 확장 + 3-vendor LLM-judge로 ROUGE 한계·self-bias를 정량 검증한 임상 노트 생성 LLM 풀스택 파이프라인.

**One-liner (EN)**: A clinical documentation LLM pipeline replicating MEDIQA-Chat 2023 (ACI-Bench Task B) with Korean extension, exposing ROUGE-Factuality decoupling (Pearson=0.254) and 3-vendor LLM-judge self-bias (0.78pt gap) through 9 quantitatively tested propositions.

> ⚠️ **연구·학습 목적.** 모든 데이터는 모의(synthetic) 시나리오. 임상 배포용 아님.

---

## ⚡ 30초 데모

```bash
git clone https://github.com/leewanho/Ambient-Clinical-Documentation-LLM.git
cd Ambient-Clinical-Documentation-LLM
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY 입력 후
streamlit run app2.py
```

→ 브라우저 자동 오픈. **5개 탭** 즉시 사용 가능:
- 🏠 **한 눈에** — Hero/Landing
- 🎯 **단일 케이스 데모** — EN/KO 동시 비교 + 3-vendor judge
- 📊 **결과 대시보드** — 9개 발견 시각화
- 🔍 **환각 사례 갤러리** — Word-level diff 색상 강조
- 🎙️ **실시간 시연** — 마이크/텍스트/파일 입력 → SOAP 노트

---

## 🚀 TL;DR — 9개 명제 정량 검증

| # | 발견 | 유형 | 결과 |
|---|---|---|---|
| 1 | gpt-4o-mini가 GPT-4(2023, WangLab) 수준 ROUGE 달성 ($0.05/40건) | 사전 가설 | ✅ |
| 2 | ROUGE는 사실성을 못 잡음 (Pearson 0.254) | 사전 가설 | ✅ |
| 3 | Korean Format이 영어 편향 judge로 부당 감점 | 사전 가설 | ❌ 폐기 |
| 4 | TF-IDF dynamic이 한국어에서 random 못 이김 | 사전 가설 | ❌ 폐기 |
| 5 | kiwi 형태소 토크나이저로도 해결 안 됨 (유사도↑ but 성능↓) | 단계 검증 | ❌ 폐기 |
| 6 | Embedding (text-emb-3-small)도 KO에서 random 못 이김 | 단계 검증 | ❌ 폐기 |
| 7 | gpt-4o vs Claude vs Gemini self-bias 0.7–0.8점 | 외부 통용 | ✅ |
| 8 | Format 점수는 judge간 합의 없음 (Pearson −0.04) | 사후 발견 | ✅ |
| 9 | KO가 EN보다 judge간 합의 높음 (Pearson 0.73 vs 0.59) | 사후 발견 | ✅ |

**분류 요약**:
- **사전 가설 4개** → 2 적중 (#1, #2) · 2 폐기 (#3, #4)
- **단계 검증 가설 2개** → 모두 폐기 (#5, #6) — 가설 폐기 자체가 발견
- **외부 통용 검증 1개** → 적중 (#7)
- **사후 패턴 발견 2개** → 적중 (#8, #9)

**가장 큰 학술적 가치** = 사전 가정 폐기 4건의 정직한 정량 입증 + 단일 LLM-judge self-bias 검증.

---

## 🛠️ Tech Stack & NLP Methods

본 프로젝트는 전통적 NLP부터 최신 생성형 AI까지 **종합 NLP 파이프라인**.

| 영역 | 사용 기술 |
|---|---|
| **Generative NLP** | gpt-4o-mini (생성), In-Context Learning, Prompt Engineering |
| **Retrieval NLP** | TF-IDF (sklearn), Dense Embedding (`text-embedding-3-small`) |
| **Korean NLP** | Kiwi 형태소 분석기 (`kiwipiepy`) |
| **Evaluation NLP** | ROUGE-1/2/L, LLM-as-judge (gpt-4o + Claude + Gemini 크로스) |
| **Speech NLP** | OpenAI Whisper ASR |
| **Clinical NLP** | ACI-Bench, MEDIQA-Chat, SOAP 노트 구조 |
| **Multilingual** | 영→한 자동 번역 + 한·영 비교 평가 |
| **Demo** | Streamlit + Plotly (5탭 인터랙티브) |

---

## 🎯 핵심 인사이트 — 3대 패러독스

### 📌 Insight 1. 메트릭의 패러독스 (The Metric Paradox)

> **"어휘가 비슷하다고 사실인 것은 아니다."**

- **발견 #2**: ROUGE × Factuality Pearson **0.254** — 약한 상관
- **발견 #6 (반전)**: Embedding이 **ROUGE-1 최고(0.5968)** 인데 **Factuality 최저(3.70)**
- **원인**: 유사 예시로 anchoring → "문장은 매끄럽지만 내용은 거짓" 환각

### 📌 Insight 2. 리트리버의 패러독스 (The Low-Resource Retriever Paradox)

> **"소규모 풀(n=67)에서는 정교한 검색이 random보다 무력하다."**
>
> *(n=67 조건 한정 — 더 큰 풀에선 결론 달라질 수 있음.)*

- **#4**: TF-IDF dynamic이 random에 패배 (KO Factuality 3.42 < 3.55)
- **#6**: Embedding도 random 못 이김 (모든 지표 하락)
- **#5**: kiwi 형태소 분석 → 유사도 ↑ but 성능 ↓ (의료 복합어 과분할)

### 📌 Insight 3. 평가자의 패러독스 (The Judge Paradox)

> **"LLM-judge는 사실엔 합의하지만 self-bias 앞에서 갈라진다."**

- **#7**: gpt-4o ↔ Gemini Factuality 격차 **0.78점** (3.83 vs 3.05) — self-bias 입증
- **#8**: Format judge간 Pearson **−0.04** → 절대 비교 불가
- **#9**: KO가 EN보다 judge 합의 높음 (Pearson 0.73 vs 0.59)

### 🛠️ Engineering Takeaway

1. **n<100 풀**: retriever 고도화보다 **고정 템플릿 Few-shot이 ROI 최고**
2. **LLM-judge 설계**: **Format 평가 제외**, **Factuality 위주 atomic checklist**
3. **Self-bias 보정**: 다중 vendor 평균 또는 normalize 공식 (Future Work)

---

## 📊 핵심 결과 (ACI-Bench test1, n=40, gpt-4o-mini ICL 2-shot)

### Retriever × Language

| Lang | Retriever | ROUGE-1 | ROUGE-L | Factuality | Completeness | Format |
|---|---|---|---|---|---|---|
| **EN** | Random | 0.5733 | 0.3300 | 3.83 | 4.03 | 4.88 |
| **EN** | TF-IDF | 0.5940 | **0.3548** | **3.98** | **4.35** | **5.00** |
| **EN** | Embedding | **0.5968** | 0.3546 | 3.70 ↓ | 4.20 | **5.00** |
| **KO** | **Random** | **0.5662** | 0.4982 | **3.55** | **3.77** | **4.92** |
| KO | TF-IDF (ws) | 0.5569 | **0.5019** | 3.42 | 3.65 | 4.88 |
| KO | TF-IDF (kiwi) | 0.5494 | 0.4993 | 3.42 | 3.70 | **4.92** |
| KO | Embedding | 0.5398 | 0.4847 | 3.50 | 3.62 | **4.92** |

### 3-Vendor Judge Self-Bias

| 지표 | gpt-4o | Claude | Gemini | 격차 |
|---|---|---|---|---|
| EN Factuality | 3.83 | 3.23 | **3.05** | **0.78점** |
| KO Factuality | 3.55 | 3.23 | **2.85** | **0.70점** |
| EN Format Pearson | — | — | — | **−0.04** (합의 없음) |

---

## 📁 데이터

| 파일 | 건수 | 출처 |
|---|---|---|
| `data/processed/aci_train.jsonl` | 67 | ACI-Bench (few-shot pool) |
| `data/processed/aci_test1.jsonl` | 40 | ACI-Bench (메인 평가) |
| `data/processed/aci_train_ko.jsonl` | 67 | gpt-4o-mini 자동번역 |
| `data/processed/aci_test1_ko.jsonl` | 40 | gpt-4o-mini 자동번역 |
| `data/processed/mts_train.jsonl` | 1,201 | MTS-Dialog (Future Work) |

---

## ⚙️ 재현 (Quick Start)

```bash
# 전처리
python src/preprocess.py

# 한국어 번역
python src/translate.py

# ICL baseline + dynamic + embedding (영·한)
python src/icl_baseline.py
python src/icl_baseline_ko.py
python src/icl_dynamic.py --lang en
python src/icl_dynamic.py --lang ko --tokenizer kiwi
python src/icl_embed.py --lang en
python src/icl_embed.py --lang ko

# LLM-as-judge (3 vendor)
python src/judge.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl
python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl \
    --provider anthropic --model claude-sonnet-4-5-20250929
python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl \
    --provider google --model gemini-2.5-flash

# 통합 분석
python src/analyze.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl

# Streamlit 5탭 데모
streamlit run app2.py
```

**전체 재현 비용**: 약 **$7** (OpenAI $3 + Anthropic $3 + Google 무료)

---

## 📂 디렉토리 구조

```
Ambient-Clinical-Documentation-LLM/
├── data/
│   ├── raw/                  # 원본 CSV (ACI-Bench, MTS-Dialog)
│   └── processed/            # JSONL 변환 결과
├── src/
│   ├── preprocess.py         # CSV → JSONL
│   ├── translate.py          # 영→한 번역
│   ├── icl_baseline.py       # EN random k-shot
│   ├── icl_baseline_ko.py    # KO random k-shot
│   ├── icl_dynamic.py        # TF-IDF dynamic (ws/kiwi)
│   ├── icl_embed.py          # OpenAI embedding
│   ├── evaluate.py           # ROUGE
│   ├── judge.py              # LLM-as-judge (gpt-4o)
│   ├── judge_multi.py        # 다중 vendor judge
│   └── analyze.py            # ROUGE + judge 통합 분석
├── notebooks/                # EDA notebooks
├── outputs/                  # 실험 결과 jsonl
├── app2.py                   # 🎙️ Streamlit 5탭 데모 (포폴 버전)
├── requirements.txt
├── .env.example
├── LIMITATIONS.md            # 상세 한계 (9 카테고리)
└── README.md
```

---

## ⚠️ 주요 한계 (TOP 5)

> 본 프로젝트는 **연구·학습용 프로토타입**. 9개 카테고리 상세 한계는 [LIMITATIONS.md](./LIMITATIONS.md) 참조.

1. **모의 시나리오**: ACI-Bench는 실 환자 대화 아닌 배우 기반 → 임상 안전 보장 불가
2. **자동 번역 한국어**: gpt-4o-mini 영→한 번역 → 실 한국 의무기록 스타일과 차이
3. **PHI/HIPAA·PIPA 미검토**: 환자 데이터 외부 API 전송 → 실 임상 적용 불가
4. **인간 의사 평가 0건**: LLM-judge는 의사 평가의 근사일 뿐, gold standard 없음
5. **n=40 단일 split**: 통계 검정·multi-seed·cross-validation 미적용

---

## 🔮 Future Work

- **인간 의사 평가** 정합성 검증 (gold standard 확보)
- **원자적 체크리스트 rubric**: 환자 정보 단위 분해 채점
- **MTS-Dialog + Pool size 단계 실험**: 67→200→500→1,200
- **의료 도메인 특화 retriever**: MedCPT, BioBERT, ClinicalBERT
- **로컬 실행 옵션**: Llama-3 + whisper.cpp (HIPAA·PIPA 대응)
- **Statistical 검증 강화**: paired t-test, bootstrap CI, multi-seed
- **Diarization 통합**: pyannote 화자 분리

---

## 📚 References

- **ACI-Bench** (Yim et al., Nature Scientific Data 2023) — [DOI](https://www.nature.com/articles/s41597-023-02487-3)
- **MEDIQA-Chat 2023 Overview** (Ben Abacha et al.) — [ACL](https://aclanthology.org/2023.clinicalnlp-1.52/)
- **WangLab at MEDIQA-Chat 2023** (우승 솔루션) — [ACL](https://aclanthology.org/2023.clinicalnlp-1.36/)
- **When Reasoning Hurts** (2026) — [arXiv](https://arxiv.org/abs/2605.24902)
- **Tierney et al.** (NEJM AI 2025) — [NEJM](https://ai.nejm.org/doi/abs/10.1056/AIoa2501000)

---

## 📄 License

코드: **MIT**. 데이터: 각 원본 출처 라이선스 준수 (ACI-Bench/MTS-Dialog).

