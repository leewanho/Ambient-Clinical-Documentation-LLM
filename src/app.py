"""
Streamlit demo: ko-medscribe-llm

3 tabs:
  1. 단일 케이스 데모 — dialogue -> note 생성 + judge 점수 (EN/KO 동시 비교)
  2. 결과 대시보드 — 9개 발견 시각화
  3. 환각 케이스 갤러리 — worst cases + judge rationale

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

st.set_page_config(page_title="ko-medscribe-llm", layout="wide", page_icon="📋")


# ---------- helpers ----------
@st.cache_data
def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def safe_judge(p):
    return {
        x["encounter_id"]: x["scores"]
        for x in load_jsonl(p)
        if isinstance(x.get("scores", {}).get("factuality"), (int, float))
    }


ICL_FILES = {
    "EN Random":         "icl_gpt-4o-mini_2shot_test1.jsonl",
    "EN TF-IDF":         "icl_dyn_gpt-4o-mini_2shot_test1.jsonl",
    "EN Embedding":      "icl_emb_gpt-4o-mini_2shot_test1.jsonl",
    "KO Random":         "icl_ko_gpt-4o-mini_2shot_n40.jsonl",
    "KO TF-IDF (ws)":    "icl_dyn_ko_gpt-4o-mini_2shot_n40.jsonl",
    "KO TF-IDF (kiwi)":  "icl_dyn_kiwi_ko_gpt-4o-mini_2shot_n40.jsonl",
    "KO Embedding":      "icl_emb_ko_gpt-4o-mini_2shot_n40.jsonl",
}

JUDGE_FILES = {
    ("EN", "gpt-4o"):  "judge_gpt-4o_icl_gpt-4o-mini_2shot_test1.jsonl",
    ("EN", "Claude"):  "judge_anthropic_claude-sonnet-4-5-20250929_icl_gpt-4o-mini_2shot_test1.jsonl",
    ("EN", "Gemini"):  "judge_google_gemini-2.5-flash_icl_gpt-4o-mini_2shot_test1.jsonl",
    ("KO", "gpt-4o"):  "judge_gpt-4o_icl_ko_gpt-4o-mini_2shot_n40.jsonl",
    ("KO", "Claude"):  "judge_anthropic_claude-sonnet-4-5-20250929_icl_ko_gpt-4o-mini_2shot_n40.jsonl",
    ("KO", "Gemini"):  "judge_google_gemini-2.5-flash_icl_ko_gpt-4o-mini_2shot_n40.jsonl",
}


def get_rouge_scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_metrics_table():
    rows = []
    for name, fname in ICL_FILES.items():
        items = load_jsonl(OUT / fname)
        if not items:
            continue
        lang = "EN" if name.startswith("EN") else "KO"
        retriever = name.split(" ", 1)[1]
        scorer = get_rouge_scorer()
        r1, rl = [], []
        for x in items:
            if not x.get("prediction"):
                continue
            sc = scorer.score(x["reference"], x["prediction"])
            r1.append(sc["rouge1"].fmeasure)
            rl.append(sc["rougeL"].fmeasure)
        judges = safe_judge(OUT / JUDGE_FILES.get((lang, "gpt-4o"), ""))
        fact = comp = fmt = None
        if judges:
            ids = [x["meta"]["encounter_id"] for x in items
                   if x.get("meta", {}).get("encounter_id") in judges]
            if ids:
                fact = mean(judges[i]["factuality"] for i in ids)
                comp = mean(judges[i]["completeness"] for i in ids)
                fmt = mean(judges[i]["format"] for i in ids)
        rows.append({
            "Language": lang,
            "Retriever": retriever,
            "N": len(r1),
            "ROUGE-1": round(mean(r1), 4) if r1 else None,
            "ROUGE-L": round(mean(rl), 4) if rl else None,
            "Factuality": round(fact, 2) if fact else None,
            "Completeness": round(comp, 2) if comp else None,
            "Format": round(fmt, 2) if fmt else None,
        })
    return pd.DataFrame(rows)


def judge_comparison_table(lang):
    g = safe_judge(OUT / JUDGE_FILES[(lang, "gpt-4o")])
    c = safe_judge(OUT / JUDGE_FILES[(lang, "Claude")])
    m = safe_judge(OUT / JUDGE_FILES[(lang, "Gemini")])
    common = sorted(set(g) & set(c) & set(m))
    rows = []
    for k in ["factuality", "completeness", "format"]:
        gv = [g[i][k] for i in common]
        cv = [c[i][k] for i in common]
        mv = [m[i][k] for i in common]
        rows.append({
            "Metric": k,
            "gpt-4o": round(mean(gv), 2) if gv else None,
            "Claude": round(mean(cv), 2) if cv else None,
            "Gemini": round(mean(mv), 2) if mv else None,
        })
    return pd.DataFrame(rows), len(common)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx * dy > 0 else 0


# ---------- header ----------
st.title("📋 ko-medscribe-llm")
st.markdown(
    "**Ambient Clinical Documentation LLM** — MEDIQA-Chat 2023 Task B 재현 + "
    "한국어 확장 + ROUGE 한계·self-bias 검증"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 단일 케이스 데모",
    "📊 결과 대시보드 (9개 발견)",
    "🔍 환각 사례 갤러리",
    "🎙️ 실시간 녹음 (Whisper)",
])


# =========================================================
# TAB 1 — single case demo (EN vs KO 동시)
# =========================================================
with tab1:
    st.header("실시간 노트 생성 + 평가 — 영·한 동시 비교")
    st.caption("같은 encounter를 영어/한국어로 동시 표시. 번역 노이즈가 사실성·포맷에 미치는 영향 직접 확인.")

    c1, c2 = st.columns([2, 2])
    with c1:
        en_items = load_jsonl(DATA / "aci_test1.jsonl")
        ko_items = load_jsonl(DATA / "aci_test1_ko.jsonl")
        if not en_items or not ko_items:
            st.error("aci_test1 / aci_test1_ko 없음 — preprocess.py + translate.py 실행 필요")
            st.stop()
        en_ids = [x.get("meta", {}).get("encounter_id", f"#{i}") for i, x in enumerate(en_items)]
        ko_ids = [x.get("meta", {}).get("encounter_id", f"#{i}") for i, x in enumerate(ko_items)]
        common_ids = sorted(set(en_ids) & set(ko_ids))
        sel = st.selectbox("케이스 ID", common_ids, key="case_t1")
    with c2:
        retriever = st.selectbox(
            "Retriever",
            ["Random", "Dynamic TF-IDF", "Embedding"],
            key="retriever_t1",
        )

    rmap = {
        ("EN", "Random"):         "icl_gpt-4o-mini_2shot_test1.jsonl",
        ("EN", "Dynamic TF-IDF"): "icl_dyn_gpt-4o-mini_2shot_test1.jsonl",
        ("EN", "Embedding"):      "icl_emb_gpt-4o-mini_2shot_test1.jsonl",
        ("KO", "Random"):         "icl_ko_gpt-4o-mini_2shot_n40.jsonl",
        ("KO", "Dynamic TF-IDF"): "icl_dyn_ko_gpt-4o-mini_2shot_n40.jsonl",
        ("KO", "Embedding"):      "icl_emb_ko_gpt-4o-mini_2shot_n40.jsonl",
    }

    en_case = en_items[en_ids.index(sel)]
    ko_case = ko_items[ko_ids.index(sel)]
    en_icl = load_jsonl(OUT / rmap[("EN", retriever)])
    ko_icl = load_jsonl(OUT / rmap[("KO", retriever)])
    en_pred = next((x for x in en_icl if x.get("meta", {}).get("encounter_id") == sel), None)
    ko_pred = next((x for x in ko_icl if x.get("meta", {}).get("encounter_id") == sel), None)

    st.divider()
    en_col, ko_col = st.columns(2)

    def render_lang(col, lang, case, pred):
        with col:
            flag = "🇺🇸" if lang == "EN" else "🇰🇷"
            st.markdown(f"### {flag} {lang}")
            st.markdown("**Dialogue (입력)**")
            st.text_area(" ", case["messages"][1]["content"], height=250,
                         label_visibility="collapsed", key=f"dlg_{lang}")
            st.markdown("**Reference (정답)**")
            st.text_area(" ", case["messages"][2]["content"], height=200,
                         label_visibility="collapsed", key=f"ref_{lang}")
            st.markdown("**Generated (모델 출력)**")
            if pred and pred.get("prediction"):
                st.text_area(" ", pred["prediction"], height=200,
                             label_visibility="collapsed", key=f"pred_{lang}")
            else:
                st.info("이 케이스의 사전 생성 결과 없음.")

    render_lang(en_col, "EN", en_case, en_pred)
    render_lang(ko_col, "KO", ko_case, ko_pred)

    st.divider()
    st.subheader("📊 3-Vendor Judge 점수 — EN vs KO 동시 비교")
    rows = []
    for lang in ["EN", "KO"]:
        for vendor in ["gpt-4o", "Claude", "Gemini"]:
            j = safe_judge(OUT / JUDGE_FILES[(lang, vendor)])
            s = j.get(sel)
            rows.append({
                "Language": lang,
                "Judge": vendor,
                "Factuality": s.get("factuality", "-") if s else "-",
                "Completeness": s.get("completeness", "-") if s else "-",
                "Format": s.get("format", "-") if s else "-",
                "Rationale (앞 100자)": (s.get("rationale", "") if s else "")[:100],
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# =========================================================
# TAB 2 — dashboard
# =========================================================
with tab2:
    st.header("결과 대시보드")
    st.markdown("ACI-Bench test1 (n=40) · gpt-4o-mini ICL · 2-shot")

    st.subheader("1. Retriever × Language 종합")
    df = compute_metrics_table()
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("2. ROUGE는 사실성을 못 잡는다 (발견 #2)")
    st.markdown("Pearson(ROUGE-1, Factuality) ≈ 0.25 — 표면 메트릭과 사실성은 약한 상관")
    scorer = get_rouge_scorer()
    icl_en = load_jsonl(OUT / "icl_gpt-4o-mini_2shot_test1.jsonl")
    j_en = safe_judge(OUT / JUDGE_FILES[("EN", "gpt-4o")])
    pts = []
    for x in icl_en:
        if not x.get("prediction"):
            continue
        eid = x["meta"].get("encounter_id")
        if eid not in j_en:
            continue
        sc = scorer.score(x["reference"], x["prediction"])
        pts.append({
            "encounter_id": eid,
            "ROUGE-1": sc["rouge1"].fmeasure,
            "Factuality": j_en[eid]["factuality"],
        })
    if pts:
        dfp = pd.DataFrame(pts)
        r = pearson(dfp["ROUGE-1"].tolist(), dfp["Factuality"].tolist())
        fig = px.scatter(dfp, x="ROUGE-1", y="Factuality",
                         hover_data=["encounter_id"],
                         title=f"Pearson r = {r:.3f}")
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Retriever 비교 — KO에서 Random이 최강 (발견 #4-6)")
    sub_df = df.dropna(subset=["Factuality"])
    fig = px.bar(sub_df, x="Retriever", y="Factuality", color="Language",
                 barmode="group", text="Factuality",
                 title="Factuality by Retriever × Language")
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(sub_df, x="Retriever", y="ROUGE-1", color="Language",
                  barmode="group", text="ROUGE-1",
                  title="ROUGE-1 by Retriever × Language")
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("4. 3-Vendor Judge — self-bias 정량 (발견 #7-9)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**EN baseline test1**")
        en_jdf, n_en = judge_comparison_table("EN")
        st.dataframe(en_jdf, hide_index=True, use_container_width=True)
        st.caption(f"n={n_en}")
    with cc2:
        st.markdown("**KO baseline test1_ko**")
        ko_jdf, n_ko = judge_comparison_table("KO")
        st.dataframe(ko_jdf, hide_index=True, use_container_width=True)
        st.caption(f"n={n_ko}")

    melt = pd.melt(
        pd.concat([en_jdf.assign(Lang="EN"), ko_jdf.assign(Lang="KO")]),
        id_vars=["Metric", "Lang"],
        value_vars=["gpt-4o", "Claude", "Gemini"],
        var_name="Judge", value_name="Score",
    )
    fig3 = px.bar(melt, x="Metric", y="Score", color="Judge", barmode="group",
                  facet_col="Lang", text="Score",
                  title="3-Vendor Judge 비교 — Format은 합의 없음 (발견 #8)")
    fig3.update_traces(textposition="outside")
    fig3.update_yaxes(range=[0, 5])
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("5. 9개 발견 요약")
    st.markdown("""
| # | 발견 | 가설 결과 |
|---|---|---|
| 1 | gpt-4o-mini가 GPT-4(2023) ROUGE 동급 ($0.05/40건) | ✅ |
| 2 | ROUGE는 사실성 못 잡음 (Pearson 0.254) | ✅ |
| 3 | ~~한국어 Format 부당 감점~~ | ❌ 폐기 |
| 4 | TF-IDF dynamic이 한국어에서 random 못 이김 | ❌ |
| 5 | kiwi 형태소 토크나이저로도 해결 안 됨 | ❌ |
| 6 | Embedding도 KO에서 random 못 이김 | ❌ |
| 7 | gpt-4o vs Claude vs Gemini self-bias 0.7-0.8점 | ✅ |
| 8 | Format 점수는 judge간 합의 없음 (Pearson −0.04) | ✅ |
| 9 | KO가 EN보다 judge간 합의 높음 (0.73 vs 0.59) | ✅ |
""")


# =========================================================
# TAB 3 — hallucination gallery
# =========================================================
with tab3:
    st.header("환각 케이스 갤러리")
    st.markdown("Factuality 낮은 순으로 정렬 · judge가 잡은 환각 패턴")

    lang_h = st.radio("언어", ["EN", "KO"], horizontal=True, key="lang_t3")
    icl_path = OUT / ("icl_ko_gpt-4o-mini_2shot_n40.jsonl" if lang_h == "KO"
                      else "icl_gpt-4o-mini_2shot_test1.jsonl")
    icl = load_jsonl(icl_path)
    judges = safe_judge(OUT / JUDGE_FILES[(lang_h, "gpt-4o")])
    j_claude = safe_judge(OUT / JUDGE_FILES[(lang_h, "Claude")])
    j_gemini = safe_judge(OUT / JUDGE_FILES[(lang_h, "Gemini")])

    rows = []
    for x in icl:
        if not x.get("prediction"):
            continue
        eid = x["meta"].get("encounter_id")
        if eid not in judges:
            continue
        rows.append({
            "id": eid,
            "fact_gpt": judges[eid]["factuality"],
            "fact_claude": j_claude.get(eid, {}).get("factuality", "-"),
            "fact_gemini": j_gemini.get(eid, {}).get("factuality", "-"),
            "rationale_gpt": judges[eid].get("rationale", ""),
            "prediction": x["prediction"],
            "reference": x["reference"],
        })
    rows.sort(key=lambda r: r["fact_gpt"])
    st.caption(f"전체 {len(rows)}건 - Factuality 낮은 순")

    top_n = st.slider("표시 개수", 1, min(20, max(1, len(rows))), 5)
    for r in rows[:top_n]:
        title = (f"{r['id']} | gpt-4o={r['fact_gpt']} "
                 f"Claude={r['fact_claude']} Gemini={r['fact_gemini']}")
        with st.expander(title):
            st.markdown(f"**Judge rationale (gpt-4o)**: _{r['rationale_gpt']}_")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Reference**")
                st.text_area("ref", r["reference"], height=300,
                             label_visibility="collapsed", key=f"ref_{r['id']}")
            with cc2:
                st.markdown("**Prediction (환각 포함)**")
                st.text_area("pred", r["prediction"], height=300,
                             label_visibility="collapsed", key=f"pred_{r['id']}")


# =========================================================
# TAB 4 — Live recording (Whisper → ICL → SOAP)
# =========================================================
with tab4:
    st.header("실시간 음성 → SOAP 노트 생성")
    st.caption("🎙️ 마이크로 의사-환자 대화를 말하면 Whisper가 텍스트화하고 ICL이 SOAP 노트를 생성")

    # 컨트롤
    lc1, lc2, lc3 = st.columns([1, 1, 2])
    with lc1:
        rec_lang = st.radio("언어", ["KO (한국어)", "EN (English)"],
                            horizontal=False, key="rec_lang")
    with lc2:
        whisper_model = st.selectbox(
            "Whisper 모델",
            ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
            help="whisper-1 = 저렴·표준 / gpt-4o-transcribe = 더 정확·고가",
        )
    with lc3:
        st.markdown("💡 **시연 팁**")
        if rec_lang.startswith("KO"):
            st.markdown(
                "> _예시 멘트:_ \"환자분, 59세 남성이고 고혈압 과거력 있으세요. "
                "며칠 전부터 가슴이 답답하고 숨이 차다고 하셨고, 운동할 때 더 심해진다고 하셨죠.\""
            )
        else:
            st.markdown(
                "> _Example:_ \"Patient is a 59-year-old male with history of hypertension. "
                "He reports chest tightness and shortness of breath for several days, worse on exertion.\""
            )

    lang_code = "ko" if rec_lang.startswith("KO") else "en"

    st.divider()

    # 입력 방법 선택: 마이크 or 파일 업로드
    input_method = st.radio(
        "입력 방법", ["🎙️ 마이크 녹음", "📁 오디오 파일 업로드"],
        horizontal=True, key="input_method",
    )

    audio_data = None
    if input_method.startswith("🎙️"):
        audio = st.audio_input("녹음 시작 → 정지", key="mic")
        if audio:
            audio_data = audio
    else:
        up = st.file_uploader(
            "오디오 파일 (mp3/wav/m4a/webm)",
            type=["mp3", "wav", "m4a", "webm", "ogg", "flac"],
            key="upload",
        )
        if up:
            audio_data = up

    # 생성 버튼
    if audio_data:
        if st.button("🔄 Transcribe & Generate SOAP", type="primary"):
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
            except ImportError:
                pass
            import os as _os
            if not _os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY가 .env에 없음")
                st.stop()

            from openai import OpenAI
            client = OpenAI()

            # 1. Transcribe
            with st.spinner(f"🎤 Whisper 전사 중 ({lang_code})..."):
                try:
                    audio_data.seek(0)
                    transcript = client.audio.transcriptions.create(
                        model=whisper_model,
                        file=("audio.wav", audio_data, "audio/wav"),
                        language=lang_code,
                    ).text
                except Exception as e:
                    st.error(f"전사 실패: {e}")
                    st.stop()

            # 2. ICL: dialogue → SOAP
            SYS_KO = ("당신은 전문 임상 AI 어시스턴트입니다. "
                      "의사와 환자 간의 대화를 바탕으로 한국 의무기록 형식의 임상 노트를 작성하세요. "
                      "다음 항목을 포함하여 작성하십시오: "
                      "주호소, 현병력, 신체검사, 검사결과, 평가 및 계획. "
                      "간결한 의무기록 서술체(~함/~임/~없음/~있음)를 사용하고 주어는 생략하십시오.")
            SYS_EN = ("You are an expert clinical AI assistant. Based on the following conversation "
                      "between a doctor and a patient, generate a structured clinical note including "
                      "CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PHYSICAL EXAMINATION, "
                      "RESULTS, ASSESSMENT AND PLAN.")
            sys_msg = SYS_KO if lang_code == "ko" else SYS_EN

            # few-shot: train pool에서 2개 랜덤
            import random as _rnd
            train_path = DATA / ("aci_train_ko.jsonl" if lang_code == "ko" else "aci_train.jsonl")
            pool = load_jsonl(train_path)
            few = _rnd.sample(pool, 2) if len(pool) >= 2 else pool

            msgs = [{"role": "system", "content": sys_msg}]
            for ex in few:
                msgs.append(ex["messages"][1])
                msgs.append(ex["messages"][2])
            user_q = f"Conversation:\n{transcript}\n\nGenerate Clinical Note:"
            msgs.append({"role": "user", "content": user_q})

            with st.spinner("📝 SOAP 노트 생성 중 (gpt-4o-mini, 2-shot)..."):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini", messages=msgs,
                        temperature=0.2, max_tokens=2048,
                    )
                    note = resp.choices[0].message.content
                except Exception as e:
                    st.error(f"노트 생성 실패: {e}")
                    st.stop()

            # 표시
            st.divider()
            col_t, col_n = st.columns(2)
            with col_t:
                st.subheader("📝 Transcript")
                st.text_area(" ", transcript, height=350,
                             label_visibility="collapsed", key="transcript_out")
            with col_n:
                st.subheader("🩺 Generated SOAP Note")
                st.text_area(" ", note, height=350,
                             label_visibility="collapsed", key="note_out")

            st.success(
                f"완료! Few-shot 예시 ID: "
                f"{[e.get('meta',{}).get('encounter_id','?') for e in few]}"
            )

    st.divider()
    with st.expander("⚠️ 한계 (정직)"):
        st.markdown("""
- **ICL pool은 자동번역 한국어**: ACI-Bench 미국 모의 대화를 gpt-4o-mini로 한국어 번역한 67건.
  실 한국 의무기록 스타일과는 차이 있음 (환자명·약물명 영문 그대로 등).
- **Whisper 의료 용어**: whisper-1은 일반 한국어는 매우 정확하나 드문 의료 용어는 가끔 오인식.
  gpt-4o-transcribe로 바꾸면 개선됨 (단, 더 비쌈).
- **짧은 녹음의 환각**: 나이·성별·기왕력 등 명시 안 하면 모델이 환각으로 채울 위험.
  데모용으론 충분하지만 임상 적용은 별도 검증 필요.
- **임상 안전 보장 없음**: 학습·연구·데모 목적. 실 환자에게 사용 금지.
""")
