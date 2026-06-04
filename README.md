# ko-medscribe-llm

**Ambient Clinical Documentation LLM** — MEDIQA-Chat 2023 (ACI-Bench Task B) 재현 + 한국어 확장 + ROUGE 한계·LLM-judge self-bias 검증.

> ⚠️ **연구·학습 목적.** 모든 데이터는 모의(synthetic) 시나리오. 임상 배포용 아님.

---

## TL;DR — 9개 발견

| # | 발견 | 결과 |
|---|---|---|
| 1 | gpt-4o-mini가 GPT-4(2023, WangLab) 수준 ROUGE 달성 ($0.05/40건) | ✅ |
| 2 | ROUGE는 사실성을 못 잡음 (Pearson 0.254) | ✅ |
| 3 | ~~Korean Format이 영어 편향 judge로 부당 감점~~ | ❌ 폐기 (실측: 차이 없음) |
| 4 | TF-IDF dynamic few-shot이 한국어에서 random 못 이김 | ❌ 가설 틀림 |
| 5 | kiwi 형태소 토크나이저로도 해결 안 됨 (유사도↑ but 성능↓) | ❌ 가설 틀림 |
| 6 | Embedding (text-emb-3-small)도 KO에서 random 못 이김 | ❌ 가설 틀림 |
| 7 | gpt-4o vs Claude vs Gemini self-bias 0.7–0.8점 | ✅ |
| 8 | Format 점수는 judge간 합의 없음 (Pearson −0.04) | ✅ |
| 9 | KO가 EN보다 judge간 합의 높음 (Pearson 0.73 vs 0.59) | ✅ |

**가설 9개 중 5개 적중, 4개 폐기.** 가장 큰 학술적 가치 = "당연한 줄 알았는데 아닌 것"을 정량 입증.

---

## 핵심 결과 표 (ACI-Bench test1, n=40, gpt-4o-mini ICL 2-shot, gpt-4o judge)

| Lang | Retriever | ROUGE-1 | ROUGE-L | Factuality | Completeness | Format |
|---|---|---|---|---|---|---|
| **EN** | Random | 0.5733 | 0.3300 | 3.83 | 4.03 | 4.88 |
| **EN** | TF-IDF | 0.5940 | **0.3548** | **3.98** | **4.35** | **5.00** |
| **EN** | Embedding | **0.5968** | 0.3546 | 3.70 ↓ | 4.20 | **5.00** |
| **KO** | **Random** | **0.5662** | 0.4982 | **3.55** | **3.77** | **4.92** |
| KO | TF-IDF (ws) | 0.5569 | **0.5019** | 3.42 | 3.65 | 4.88 |
| KO | TF-IDF (kiwi) | 0.5494 | 0.4993 | 3.42 | 3.70 | **4.92** |
| KO | Embedding | 0.5398 | 0.4847 | 3.50 | 3.62 | **4.92** |

**관찰**:
- 영어: Embedding이 ROUGE-1 최고지만 Factuality 최저 → 발견 #2의 결정적 증거 (표면 메트릭과 사실성의 디커플)
- 한국어: Random이 모든 지표 1위 → 발견 #4-6 (의료 도메인 ICL에서 다양성이 유사도보다 강함)

## 3-Vendor Judge 비교 (self-bias 입증)

### EN (test1, n=40)
| 지표 | gpt-4o | Claude | Gemini | 격차 |
|---|---|---|---|---|
| Factuality | 3.83 | 3.23 | **3.05** | **0.78점** |
| Completeness | 4.03 | 4.05 | 3.58 | 0.47점 |
| Format | 4.88 | 4.95 | 4.65 | 0.30점 |

### KO (test1_ko, n=40)
| 지표 | gpt-4o | Claude | Gemini | 격차 |
|---|---|---|---|---|
| Factuality | 3.55 | 3.23 | **2.85** | **0.70점** |
| Completeness | 3.77 | 3.42 | 2.88 | 0.89점 |
| Format | 4.92 | 4.47 | 4.35 | 0.57점 |

**Inter-rater Pearson (3-judge 평균)**
| 지표 | EN | KO |
|---|---|---|
| Factuality | 0.59 | **0.73** |
| Completeness | 0.40 | 0.65 |
| Format | **−0.04** | 0.29 |

→ Factuality는 강한 합의, **Format은 합의 없음** (절대값 비교 불가).

---

## 데이터

| 파일 | 건수 | 출처 | 본 프로젝트 사용 |
|---|---|---|---|
| `data/processed/aci_train.jsonl` | 67 | ACI-Bench | ✓ few-shot pool |
| `data/processed/aci_valid.jsonl` | 20 | ACI-Bench | ✓ dry run |
| `data/processed/aci_test1.jsonl` | 40 | ACI-Bench | ✓ 메인 평가 |
| `data/processed/aci_train_ko.jsonl` | 67 | ACI-Bench (gpt-4o-mini 자동번역) | ✓ KO few-shot pool |
| `data/processed/aci_test1_ko.jsonl` | 40 | ACI-Bench (자동번역) | ✓ KO 메인 평가 |
| `data/processed/mts_train.jsonl` | 1,201 | MTS-Dialog | △ 전처리만, Future Work |
| `data/processed/mts_valid.jsonl` | 100 | MTS-Dialog | △ 전처리만, Future Work |

원본 CSV는 `data/raw/`. `python src/preprocess.py`로 재생성 가능.

---

## 셋업

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 열어서 API 키 입력:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...   (선택, judge_multi.py용)
#   GOOGLE_API_KEY=...             (선택, judge_multi.py용)
```

## 재현 (Quick Start)

```bash
# 1. 전처리 (CSV → JSONL)
python src/preprocess.py

# 2. ICL 베이스라인 (영어)
python src/icl_baseline.py                       # n=20 (~$0.025)
python src/evaluate.py outputs/icl_gpt-4o-mini_2shot_n20.jsonl

# 3. ICL test1 (영어, n=40, ~$0.05)
python src/icl_baseline.py \
    --input data/processed/aci_test1.jsonl \
    --output outputs/icl_gpt-4o-mini_2shot_test1.jsonl

# 4. 한국어 번역 (~$0.50)
python src/translate.py

# 5. ICL 한국어 (~$0.05)
python src/icl_baseline_ko.py

# 6. Dynamic few-shot (TF-IDF / kiwi / embedding)
python src/icl_dynamic.py --lang en
python src/icl_dynamic.py --lang ko --tokenizer kiwi
python src/icl_embed.py --lang en
python src/icl_embed.py --lang ko

# 7. LLM-as-judge (gpt-4o)
python src/judge.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl

# 8. 다중 judge (self-bias 검증)
python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl \
    --provider anthropic --model claude-sonnet-4-5-20250929
python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl \
    --provider google --model gemini-2.5-flash

# 9. 통합 분석 (ROUGE + judge 상관)
python src/analyze.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl

# 10. Streamlit 데모 (브라우저 자동 오픈)
streamlit run src/app.py
```

**전체 재현 비용**: 약 $7 (OpenAI $3 + Anthropic $3 + Google 무료)

---

## 디렉토리 구조

```
ko-medscribe-llm/
├── data/
│   ├── raw/                  # 원본 CSV (ACI-Bench, MTS-Dialog)
│   └── processed/            # JSONL 변환 결과
├── src/
│   ├── preprocess.py         # CSV → JSONL
│   ├── translate.py          # 영→한 번역
│   ├── icl_baseline.py       # EN random k-shot ICL
│   ├── icl_baseline_ko.py    # KO random k-shot ICL
│   ├── icl_dynamic.py        # TF-IDF dynamic (whitespace/kiwi)
│   ├── icl_embed.py          # OpenAI embedding dynamic
│   ├── evaluate.py           # ROUGE
│   ├── judge.py              # LLM-as-judge (gpt-4o)
│   ├── judge_multi.py        # 다중 vendor judge
│   ├── analyze.py            # ROUGE + judge 통합 분석
│   └── app.py                # Streamlit 데모
├── outputs/                  # 실험 결과 jsonl
├── requirements.txt
├── .env.example
└── README.md
```

---

## 데모 (Streamlit)

```bash
streamlit run src/app.py
```

3개 탭:
1. **단일 케이스 데모** — 같은 encounter의 EN/KO 동시 표시 + 3-vendor judge 점수
2. **결과 대시보드** — 9개 발견 시각화 (ROUGE vs Factuality 산점도, retriever 비교, judge 비교)
3. **환각 사례 갤러리** — Factuality 낮은 케이스 + 판단 근거

---

## 한계 및 Future Work

**한계**:
- 데이터 = 모의 시나리오 (ACI-Bench도 실 환자 아님). **임상 안전 보장 못 함.**
- 한국어 = gpt-4o-mini 자동번역 → 노이즈 포함. 실 한국 임상 데이터로 검증 필요.
- n=40는 통계적 유의성 한계. 신뢰구간 좁히려면 더 큰 평가셋 필요.
- LLM-as-judge는 인간 의사 평가의 근사일 뿐.

**Future Work**:
- MTS-Dialog Task A (섹션 단위) 실험 추가
- 의료 도메인 특화 retriever (BioBERT/MedCLIP embedding)
- LoRA fine-tuning 비교
- 실 한국 의료 대화 코퍼스 확보 (IRB 필요)
- 인간 의사 평가 + LLM-judge 정합성 분석

---

## 참고문헌

- **ACI-Bench** (Yim et al., Nature Scientific Data 2023) — https://www.nature.com/articles/s41597-023-02487-3
- **MEDIQA-Chat 2023 Overview** (Ben Abacha et al.) — https://aclanthology.org/2023.clinicalnlp-1.52/
- **WangLab at MEDIQA-Chat 2023** (winning solution, GPT-4 ICL) — https://aclanthology.org/2023.clinicalnlp-1.36/
- **When Reasoning Hurts** (2026) — https://arxiv.org/abs/2605.24902
- **Tierney et al., NEJM AI 2025** (Ambient AI scribe RCT) — https://ai.nejm.org/doi/abs/10.1056/AIoa2501000

---

## 라이선스

코드: MIT. 데이터: 각 원본 출처의 라이선스 준수 (ACI-Bench/MTS-Dialog).
