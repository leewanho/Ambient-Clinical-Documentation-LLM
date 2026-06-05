# Ambient Clinical Documentation LLM

**Ambient Clinical Documentation LLM** — MEDIQA-Chat 2023 (ACI-Bench Task B) 재현 + 한국어 확장 + ROUGE 한계·LLM-judge self-bias 검증.

> ⚠️ **연구·학습 목적.** 모든 데이터는 모의(synthetic) 시나리오. 임상 배포용 아님.

---

## TL;DR — 9개 발견

| # | 발견 | 유형 | 결과 |
|---|---|---|---|
| 1 | gpt-4o-mini가 GPT-4(2023, WangLab) 수준 ROUGE 달성 ($0.05/40건) | 사전 가설 | ✅ 가설 적중 |
| 2 | ROUGE는 사실성을 못 잡음 (Pearson 0.254) | 사전 가설 | ✅ 가설 적중 |
| 3 | Korean Format이 영어 편향 judge로 부당 감점 | 사전 가설 | ❌ 가설 폐기 |
| 4 | TF-IDF dynamic few-shot이 한국어에서 random 못 이김 | 사전 가설 | ❌ 가설 폐기 |
| 5 | kiwi 형태소 토크나이저로도 해결 안 됨 (유사도↑ but 성능↓) | 단계 검증 가설 | ❌ 가설 폐기 |
| 6 | Embedding (text-emb-3-small)도 KO에서 random 못 이김 | 단계 검증 가설 | ❌ 가설 폐기 |
| 7 | gpt-4o vs Claude vs Gemini self-bias 0.7–0.8점 | 외부 통용 검증 | ✅ 가설 적중 |
| 8 | Format 점수는 judge간 합의 없음 (Pearson −0.04) | 사후 패턴 발견 | ✅ 가설 적중 |
| 9 | KO가 EN보다 judge간 합의 높음 (Pearson 0.73 vs 0.59) | 사후 패턴 발견 | ✅ 가설 적중 |

**9개 명제 정량 검증 요약**:
- **사전 가설 4개** → 2 적중 (#1, #2) · 2 폐기 (#3, #4)
- **단계 검증 가설 2개** → 모두 폐기 (#5, #6) — 가설 폐기 자체가 발견
- **외부 통용 검증 1개** → 적중 (#7)
- **사후 패턴 발견 2개** → 적중 (#8, #9)

가장 큰 학술적 가치 = 사전 가정 폐기 4건의 정직한 정량 입증 + 단일 LLM-judge self-bias 검증.

---

## 🛠️ Tech Stack & NLP Methods (사용한 NLP 기술)

본 프로젝트는 전통적인 자연어 처리(NLP) 기법부터 최신 생성형 AI(Generative AI) 기술을 아우르는 **종합 NLP 파이프라인**을 갖추고 실험·구현되었습니다.

- **Generative NLP (생성 및 추론)**: In-Context Learning (Few-shot ICL), Prompt Engineering, `gpt-4o-mini` (생성 모델)
- **Retrieval NLP (정보 검색)**: TF-IDF (Term Frequency-Inverse Document Frequency) Retriever, Dense Text Embedding Retriever (`text-embedding-3-small`)
- **Korean Tokenization (한국어 형태소 분석)**: Kiwi 형태소 분석기 (`kiwipiepy`) 기반의 토크나이저 및 어휘 유사도 검색
- **Evaluation NLP (자연어 평가)**: ROUGE-1 / ROUGE-2 / ROUGE-L (어휘 중첩도 기반 자동 평가), LLM-as-a-Judge 크로스 평가 (gpt-4o, Claude, Gemini)
- **Speech NLP (음성 언어 처리)**: OpenAI Whisper ASR (자동 음성 인식) API를 통한 실시간 의사-환자 음성 전사 및 요약 파이프라인

---

## 🚀 핵심 인사이트 — 3대 패러독스

위 9개 발견을 가로지르는 **3가지 역설(Paradox)** 로 본 프로젝트의 진짜 가치를 정리.

### 📌 Insight 1. 메트릭의 패러독스 (The Metric Paradox)

> **"어휘가 비슷하다고 사실인 것은 아니다."**
> 자연어 처리 표준 메트릭(ROUGE)이 의료 도메인에서 갖는 한계와 위험성.

**🔍 어휘적 유사도와 임상적 사실성의 디커플링**
- **발견 #2**: ROUGE와 LLM-judge Factuality 간 Pearson 상관계수 **0.254** — 약한 상관. ROUGE는 임상적 팩트의 왜곡(나이·이름·약물명 환각)을 잡아내지 못함.

**🔍 검색 고도화가 유발한 "그럴듯한 환각"**
- **발견 #6 (영어 데이터의 반전)**: Embedding 검색(`text-embedding-3-small`)이 **ROUGE-1 최고점(0.5968)** 을 기록했으나, 정작 **Factuality는 최저치(3.70 ↓)** 로 추락.
- **원인**: 리트리버가 어휘적으로 가장 유사한 Few-shot 예시를 가져오니, 모델이 그 예시의 다른 환자 진단명·약물 맥락에 오염(Contamination)되어 "문장은 매끄럽지만 내용은 거짓인" 환각을 생성.

---

### 📌 Insight 2. 리트리버의 패러독스 (The Low-Resource Retriever Paradox)

> **"소규모 도메인 풀(n=67)에서는 정교한 검색 알고리즘이 무작위(Random)보다 무력하다."**
> RAG/고도화된 퓨샷 검색이 항상 우수할 것이라는 공학적 가정에 대한 반례. **(단, n=67 조건 한정 — 더 큰 풀에선 결론 달라질 수 있음. Future Work 참조.)**

**🔍 한국어 환경에서 무력화된 TF-IDF·Embedding**
- **발견 #4**: TF-IDF Dynamic Few-shot이 Random을 못 이김 (Factuality: Random 3.55 vs TF-IDF 3.42).
- **발견 #6**: Multilingual Embedding 검색도 Random 대비 우위 없음 — 모든 지표 하락 (Factuality 3.50, ROUGE-1 0.5398).

**🔍 도메인 특성이 결여된 토크나이저의 역효과**
- **발견 #5**: 일반 한국어 형태소 분석기(`kiwi`)로 어휘 유사도를 강제로 높였으나(0.21→0.35), 생성 성능은 오히려 저하. 의료 복합어("고혈압" → "고"+"혈압") 과분할로 리트리버가 엉뚱한 퓨샷 매칭.

---

### 📌 Insight 3. 평가자의 패러독스 (The Judge Paradox)

> **"LLM-judge는 사실을 판별할 땐 협력하지만, 자존심(Self-bias) 앞에서는 갈라진다."**
> 인간 평가가 없는 상황에서 LLM-as-judge 방법론을 사용할 때의 신뢰 경계선.

**🔍 3대 대형 LLM의 정량적 Self-bias**
- **발견 #7**: `gpt-4o` / `Claude` / `Gemini` 3개 모델 크로스 판정 결과, 자신과 같은 vendor(`gpt-4o-mini`) 출력에 **평균 0.7~0.8점 가산점**. Factuality 격차: gpt-4o(3.83) vs Gemini(3.05) — 0.78점.

**🔍 포맷(Format) 평가의 무용성 vs 사실성의 합의**
- **발견 #8**: Format 점수 Judge간 Pearson 상관 **−0.04** — 상호 합의 전무. **LLM-judge에게 절대적 포맷 채점은 불가능.**
- **발견 #9**: 반대로 Factuality는 한국어에서 EN(0.59)보다 KO(0.73) judge간 합의 더 높음. 번역 노이즈로 오류가 모든 judge에 명백.

---

### 🛠️ Engineering Takeaway

1. **엔지니어링 낭비 방지**: 소규모 데이터 풀(n<100)에서는 리트리버 고도화(Embedding, 형태소 분석)에 리소스 쓰기보다 **확실한 고정 템플릿 Few-shot이 성능·비용 모두 이득**.
2. **LLM 평가 가이드라인**: LLM-judge 설계 시 **Format 평가는 제외**, **Factuality 위주의 원자적(Atomic) 체크리스트** 권장. Self-bias 보정은 Future Work — 다중 vendor 평균 또는 normalize 공식 필요.
3. **실시간 데모로 검증**: `streamlit run app2.py` 4개 탭에서 위 3대 패러독스 + 환각 사례 갤러리 시각화.

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
- 영어: Embedding이 ROUGE-1 최고지만 Factuality 최저 → 발견 #2의 결정적 증거
- 한국어: Random이 모든 지표 1위 → 발견 #4-6

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

---

## 셋업

```bash
pip install -r requirements.txt
cp .env.example .env
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...   (선택)
# GOOGLE_API_KEY=...             (선택)
```

## 재현 (Quick Start)

```bash
python src/preprocess.py
python src/icl_baseline.py
python src/translate.py
python src/icl_baseline_ko.py
python src/icl_dynamic.py --lang en
python src/icl_dynamic.py --lang ko --tokenizer kiwi
python src/icl_embed.py --lang en
python src/icl_embed.py --lang ko
python src/judge.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl
python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl --provider anthropic --model claude-sonnet-4-5-20250929
python src/judge_multi.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl --provider google --model gemini-2.5-flash
python src/analyze.py outputs/icl_gpt-4o-mini_2shot_test1.jsonl
streamlit run src/app.py
```

**전체 재현 비용**: 약 $7

---

## 디렉토리 구조

```
ko-medscribe-llm/
├── data/
│   ├── raw/                  # 원본 CSV
│   └── processed/            # JSONL
├── src/
│   ├── preprocess.py · translate.py
│   ├── icl_baseline.py · icl_baseline_ko.py
│   ├── icl_dynamic.py · icl_embed.py
│   ├── evaluate.py · judge.py · judge_multi.py
│   ├── analyze.py
│   └── app.py                # Streamlit 데모 (4 tabs)
├── outputs/
├── requirements.txt · .env.example · README.md
```

---

## 데모 (Streamlit)

```bash
streamlit run src/app.py
```

4개 탭:
1. **단일 케이스 데모** — EN/KO 동시 + 3-vendor judge 점수
2. **결과 대시보드** — 9개 발견 시각화
3. **환각 사례 갤러리** — Factuality 낮은 케이스 + 판단 근거
4. **🎙️ 실시간 녹음 (Whisper)** — 마이크 → 한·영 SOAP 노트 생성

---

## 한계 (Limitations)

본 프로젝트는 **연구·학습용 프로토타입**. 실 임상 적용 전 반드시 검토할 한계.

### 1. 데이터
- **모의 시나리오**: ACI-Bench는 배우 기반 모의 대화. 실 환자-의사 자연 대화의 머뭇거림·중단·동시 발화 부재.
- **미국 의료 환경 종속**: CPT 코드, 미국 약물명, 미국 보험 체계 기반 → 한국 진료 패턴·약가·심사 기준 반영 불가.
- **샘플 크기**: train 67 / test 40 — fine-tuning 부족, statistical power 한계.
- **테스트 split 단일**: 1개 split만 평가 → variance 통제 불가, cross-validation 미적용.
- **음성 원본 부재**: 이미 텍스트화된 dialogue만. 실 음성 노이즈·diarization 효과 미반영.
- **메타데이터 분리**: 환자 나이·성별·기왕력이 별도 metadata CSV에 있고 dialogue엔 자연어로만 등장 → 모델이 환자 식별 정보를 환각으로 만들기 쉬움.
- **데이터 누수 가능성**: ACI-Bench는 2023년 공개 → gpt-4o-mini 학습 데이터에 포함됐을 가능성. 본 결과의 과대평가 위험 미검증.

### 2. 한국어 파이프라인
- **자동 번역 노이즈**: gpt-4o-mini 영→한 번역. 의료 용어 정확도 미검증.
- **번역체 vs 실 의무기록**: 환자명·약물명 영문 유지, 한국 의무기록 관습(약어, KCD-7) 미반영.
- **한국 표준 미연계**: 보건복지부 의무기록 가이드, KCD-7, KAAACI 가이드라인 미사용.
- **한국 의료 어휘 사전 부재**: kiwi는 일반어 형태소 분석기. 의료 복합어("고혈압" → "고"+"혈압") 과분할.

### 3. 모델·인프라
- **Closed-source 의존**: gpt-4o-mini / gpt-4o / Claude / Gemini / Whisper 모두 외부 API. 모델 silent update 시 재현성 흔들림.
- **PHI/개인정보 처리 부재**: 환자 데이터가 OpenAI·Anthropic·Google 서버로 전송. **HIPAA·PIPA 준수 미검토**.
- **로컬 실행 옵션 없음**: Llama·whisper.cpp 등 self-hosted 미구현.
- **모델 버전 미고정**: requirements.txt에 model snapshot pin 없음.

### 4. ICL·Retriever 실험
- **k=2만 테스트**: k=4, 8 효과 미검증.
- **Pool 크기 67 고정**: random 우위가 큰 풀(MTS 1,201)에서도 유지될지 미검증.
- **Temperature 0.2 고정**: 0.0, 0.5 미탐색.
- **Embedding 1종**: text-embedding-3-small만. multilingual-e5, BGE-M3, MedCPT, BioBERT 미시도.
- **Cross-encoder 재랭킹 미시도**.
- **Hybrid retrieval (BM25+dense) 미시도**.
- **Fine-tuning 비교 없음**.

### 5. 평가 방법론
- **인간 의사 평가 0건**: LLM-as-judge는 근사. Gold standard 없음.
- **의료 특화 메트릭 부재**: MEDCON, UMLS entity F1, ICD coding accuracy 미사용.
- **통계 검정 부재**: paired t-test, McNemar, bootstrap CI 안 함 — retriever 차이가 우연인지 미검증.
- **단일 seed**: 1회 실행만. multi-seed 평균·분산 없음.
- **Inter-rater agreement 부분 측정**: Pearson만. Cohen's kappa·Krippendorff α 미보고.
- **Judge 자체 환각**: judge의 평가가 항상 옳다는 가정 미검증.

### 6. Whisper / ASR
- **의료 전문용어 인식률 미검증**: 한국어 약물명·진단명 정확도 미측정.
- **Diarization 부재**: 의사·환자 화자 분리 없음 → 누가 말했는지 모델이 추론.
- **실시간 스트리밍 X**: 녹음 종료 후 일괄 전사.
- **잡음·다중 화자·마스크 환경 미평가**.

### 7. 임상 안전·규제
- **임상 검증 0건**: 의사·환자 안전 평가 부재.
- **약물 상호작용·금기 체크 없음**.
- **표준 코딩(ICD-10/SNOMED CT/KCD-7) 미연계**.
- **의료기기 규제 미검토**: 식약처 SaMD, FDA 510(k).
- **감사 로그·접근 통제·비식별화 없음**.
- **의사 승인·수정 워크플로우 부재**.

### 8. 엔지니어링
- **단위 테스트 0개**.
- **CI/CD 미구성**.
- **에러 처리 최소화**: API 실패 시 단순 skip, retry/backoff 미구현.
- **로깅 시스템 없음**: print만.
- **비동기 호출 미사용**: 대량 처리 시 비효율.
- **Streamlit 데모 미배포**: 로컬만, 동시 사용자·인증·rate limit 없음.

### 9. 본 프로젝트 자체의 메타 한계
- **Random 채택의 위험**: KO에서 random이 최강이라는 발견은 "통제 불가능한 변동성" — 상용 환경에선 매 실행 결과 다름.
- **Retriever 비교에 ROUGE 사용**: ROUGE 한계를 입증한 프로젝트가 architecture 선택엔 ROUGE를 부분 참고 — circular 한계.
- **Format 메트릭 자체 의문**: judge간 합의 음수(−0.04) → 본 프로젝트 Format 점수는 사실상 의미 없음.

---

## Future Work

- **인간 의사 평가**와 LLM-judge 정합성 분석.
- **원자적(Atomic) 체크리스트 rubric**: 환자 나이/성별/약물/진단 단위로 분해 채점.
- **MTS-Dialog Task A + Pool size 단계 실험**: 67→200→500→1,200 retriever 효용 곡선.
- **한국어 실 임상 코퍼스**: IRB·비식별화 후 fine-tuning 비교.
- **의료 도메인 특화 retriever**: MedCPT, BioBERT, ClinicalBERT.
- **Whisper 의료 어휘 후처리**: 한국어 의료 사전 기반 오인식 보정.
- **로컬 실행 옵션**: Llama-3-8B + whisper.cpp (HIPAA·PIPA 대응).
- **Statistical 검증 강화**: paired test, bootstrap CI, multi-seed.
- **Diarization 통합**: pyannote 화자 분리.

---

## 참고문헌

- **ACI-Bench** (Yim et al., Nature Scientific Data 2023) — https://www.nature.com/articles/s41597-023-02487-3
- **MEDIQA-Chat 2023 Overview** (Ben Abacha et al.) — https://aclanthology.org/2023.clinicalnlp-1.52/
- **WangLab at MEDIQA-Chat 2023** — https://aclanthology.org/2023.clinicalnlp-1.36/
- **When Reasoning Hurts** (2026) — https://arxiv.org/abs/2605.24902
- **Tierney et al., NEJM AI 2025** — https://ai.nejm.org/doi/abs/10.1056/AIoa2501000

---

## 라이선스

코드: MIT. 데이터: 각 원본 출처 라이선스 준수 (ACI-Bench/MTS-Dialog).
