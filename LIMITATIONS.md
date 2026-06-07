# Limitations — 상세 한계

본 프로젝트는 **연구·학습용 프로토타입**입니다. 실 임상 적용 전 반드시 검토해야 할 한계를 9개 영역으로 분류해 정리했습니다.

핵심 5가지는 [README의 주요 한계 섹션](./README.md#-주요-한계-top-5)에서 확인 가능.

---

## 1. 데이터

- **모의 시나리오**: ACI-Bench는 배우 기반 모의 대화. 실 환자-의사 자연 대화의 머뭇거림·중단·동시 발화·감정 표현 부재.
- **미국 의료 환경 종속**: CPT 코드, 미국 약물명, 미국 보험 체계 기반 → 한국 진료 패턴·약가·심사 기준 반영 불가.
- **샘플 크기**: train 67 / test 40 — fine-tuning 부족, statistical power 한계.
- **테스트 split 단일**: 1개 split만 평가 → variance 통제 불가, cross-validation 미적용.
- **음성 원본 부재**: 이미 텍스트화된 dialogue만. 실 음성 노이즈·diarization 효과 미반영.
- **메타데이터 분리**: 환자 나이·성별·기왕력이 별도 metadata CSV에 있고 dialogue엔 자연어로만 등장 → 모델이 환자 식별 정보를 환각으로 만들기 쉬움.
- **데이터 누수 가능성**: ACI-Bench는 2023년 공개 → gpt-4o-mini 학습 데이터에 포함됐을 가능성. 본 결과의 과대평가 위험 미검증.

## 2. 한국어 파이프라인

- **자동 번역 노이즈**: gpt-4o-mini 영→한 번역 67+40건. 의료 용어 정확도 미검증.
- **번역체 vs 실 의무기록**: 환자명·약물명 영문 유지, 한국 의무기록 관습(약어, KCD-7 코딩) 미반영.
- **한국 표준 미연계**: 보건복지부 의무기록 가이드라인, KCD-7, KAAACI 가이드라인 미사용.
- **한국 의료 어휘 사전 부재**: kiwi는 일반어 형태소 분석기. 의료 복합어("고혈압" → "고"+"혈압") 과분할.

## 3. 모델·인프라

- **Closed-source 의존**: gpt-4o-mini / gpt-4o / Claude / Gemini / Whisper 모두 외부 API. 모델 silent update 시 재현성 흔들림.
- **PHI/개인정보 처리 부재**: 환자 데이터가 OpenAI·Anthropic·Google 서버로 전송. **HIPAA(미국)·PIPA(한국) 준수 미검토**.
- **로컬 실행 옵션 없음**: Llama·whisper.cpp 등 self-hosted 미구현.
- **모델 버전 미고정**: `gpt-4o-mini`는 OpenAI가 silent update 가능. requirements.txt에 정확한 snapshot pin 미적용.

## 4. ICL·Retriever 실험

- **k=2만 테스트**: k=4, 8 등 다른 shot 수 효과 미검증.
- **Pool 크기 67 고정**: 작은 풀에서 random이 유리하다는 발견이 큰 풀(MTS 1,201)에서도 유지될지 미검증.
- **Temperature 0.2 고정**: 다른 값(0.0, 0.5) 미탐색.
- **Embedding 1종**: OpenAI text-embedding-3-small만 사용. multilingual-e5, BGE-M3, 의료 특화 MedCPT/BioBERT 미시도.
- **Cross-encoder 재랭킹 미시도**: 1-stage retrieval만.
- **Hybrid retrieval (BM25+dense) 미시도**.
- **Fine-tuning 비교 없음**: WangLab도 같은 결론(ICL > FT)이지만 본 프로젝트에서 직접 입증 안 함.

## 5. 평가 방법론

- **인간 의사 평가 0건**: LLM-as-judge는 의사 평가의 근사일 뿐. Gold standard 없음.
- **의료 특화 메트릭 부재**: MEDCON, UMLS entity F1, ICD coding accuracy 등 미사용.
- **표면 메트릭 의존**: ROUGE/BERTScore 외 의미·임상 안전 평가 부족.
- **통계적 유의성 검정 부재**: paired t-test, McNemar, bootstrap CI 등 안 함 — retriever 간 차이가 우연인지 검증 안 됨.
- **단일 seed**: 1회 실행 결과만 보고. multi-seed 평균·분산 없음.
- **Inter-rater agreement 부분 측정**: Pearson만 사용. Cohen's kappa·Krippendorff α 같은 더 엄격한 ordinal agreement 메트릭 미보고.
- **Judge 자체 환각**: judge 모델의 평가가 항상 옳다는 가정 검증 안 됨.

## 6. Whisper / ASR (실시간 녹음 탭)

- **의료 전문용어 인식률 미검증**: 한국어 의료 어휘(약물명, 진단명) 정확도 측정 안 함.
- **Diarization 부재**: 의사·환자 화자 분리 안 됨 → 누가 말했는지 모델이 추론으로 채움.
- **실시간 스트리밍 X**: 녹음 종료 후 일괄 전사.
- **잡음·다중 화자·마스크 환경 미평가**.

## 7. 임상 안전·규제

- **임상 검증 0건**: 의사·환자 안전 평가 부재.
- **약물 상호작용·금기 체크 없음**.
- **표준 코딩(ICD-10/SNOMED CT/KCD-7) 미연계**.
- **의료기기 규제 미검토**: 한국 식약처 SaMD, 미국 FDA 510(k) 등.
- **감사 로그·접근 통제·비식별화 없음**.
- **의사 승인·수정 워크플로우 부재**.

## 8. 엔지니어링

- **단위 테스트 0개**.
- **CI/CD 미구성**.
- **에러 처리 최소화**: API 실패 시 단순 skip, retry/backoff 미구현.
- **로깅 시스템 없음**: print만 사용.
- **비동기 호출 미사용**: 대량 처리 시 비효율.
- **Streamlit 데모 미배포**: 로컬만, 동시 사용자·인증·rate limit 없음.

## 9. 본 프로젝트 자체의 메타 한계

- **Random 채택의 위험**: KO에서 random이 최강이라는 발견은 "통제된 변동성"이 아닌 "통제 불가능한 변동성" — 상용 환경에선 매 실행 결과 다름.
- **Retriever 비교에 ROUGE 사용**: ROUGE의 한계를 입증한 같은 프로젝트가 architecture 선택엔 ROUGE를 부분적으로 참고 — circular 한계.
- **Format 메트릭 사용 자체 의문**: judge간 합의 음수(−0.04) → 본 프로젝트에서 보고한 Format 점수는 사실상 의미 없음.

---

← [README로 돌아가기](./README.md)
