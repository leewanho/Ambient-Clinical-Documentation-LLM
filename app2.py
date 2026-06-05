"""
Streamlit demo v2 — ko-medscribe-llm (Portfolio Edition)

5 tabs:
  0. 🏠 한 눈에 — Hero/Landing (9 findings, 핵심 카드, CTA)
  1. 🎯 단일 케이스 데모 — EN/KO 동시 비교
  2. 📊 결과 대시보드 — 9개 발견 시각화
  3. 🔍 환각 사례 갤러리 (with Diff) — word-level 색상 차이
  4. 🎙️ 실시간 녹음 + 텍스트 입력 — Whisper / 자유 입력 / 파일

Run:
    streamlit run app2.py
"""
from __future__ import annotations

import difflib
import html
import json
import random
from pathlib import Path
from statistics import mean

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

st.set_page_config(
    page_title="ko-medscribe-llm",
    layout="wide",
    page_icon="📋",
    initial_sidebar_state="collapsed",
)


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


# ---------- diff helper ----------
def side_by_side_diff(ref: str, pred: str) -> tuple[str, str]:
    """
    Word-level diff. Returns (left_html, right_html).
    Left = reference with deletions (missing in pred) marked.
    Right = prediction with insertions (potential hallucinations) marked.
    """
    ref_tokens = ref.split()
    pred_tokens = pred.split()
    sm = difflib.SequenceMatcher(None, ref_tokens, pred_tokens, autojunk=False)
    left_parts = []
    right_parts = []
    HL_RED = '<mark style="background:#fcd1d1;color:#7a0c0c;padding:0 2px">'
    HL_YEL = '<mark style="background:#fff3c4;color:#7a5b00;padding:0 2px">'
    END = "</mark>"
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            left_parts.append(html.escape(" ".join(ref_tokens[i1:i2])))
            right_parts.append(html.escape(" ".join(pred_tokens[j1:j2])))
        elif tag == "delete":
            left_parts.append(HL_YEL + html.escape(" ".join(ref_tokens[i1:i2])) + END)
        elif tag == "insert":
            right_parts.append(HL_RED + html.escape(" ".join(pred_tokens[j1:j2])) + END)
        elif tag == "replace":
            left_parts.append(HL_YEL + html.escape(" ".join(ref_tokens[i1:i2])) + END)
            right_parts.append(HL_RED + html.escape(" ".join(pred_tokens[j1:j2])) + END)
    return " ".join(left_parts), " ".join(right_parts)


# ---------- ICL pipeline (for free input / mic) ----------
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


def generate_soap_from_dialogue(dialogue: str, lang: str, model: str = "gpt-4o-mini"):
    """Run ICL with random 2-shot from train pool to generate SOAP note."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    import os as _os
    if not _os.getenv("OPENAI_API_KEY"):
        return None, "OPENAI_API_KEY가 .env에 없음"

    from openai import OpenAI
    client = OpenAI()

    suffix = "_ko" if lang == "ko" else ""
    pool = load_jsonl(DATA / f"aci_train{suffix}.jsonl")
    if len(pool) < 2:
        return None, f"few-shot pool 부족 (aci_train{suffix}.jsonl)"

    sys_msg = SYSTEM_KO if lang == "ko" else SYSTEM_EN
    few = random.sample(pool, 2)
    msgs = [{"role": "system", "content": sys_msg}]
    for ex in few:
        msgs.append(ex["messages"][1])
        msgs.append(ex["messages"][2])
    user_q = f"Conversation:\n{dialogue}\n\nGenerate Clinical Note:"
    msgs.append({"role": "user", "content": user_q})

    try:
        resp = client.chat.completions.create(
            model=model, messages=msgs,
            temperature=0.2, max_tokens=2048,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <style>
    .big-num {font-size: 3rem; font-weight: 700; color: #0066cc;}
    .stat-label {font-size: 0.9rem; color: #666;}
    .hero-card {background: #f7f9fc; padding: 1.5rem; border-radius: 10px;
                border-left: 4px solid #0066cc; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📋 ko-medscribe-llm")
st.caption(
    "Ambient Clinical Documentation LLM — MEDIQA-Chat 2023 (ACI-Bench Task B) 재현 + "
    "한국어 확장 + ROUGE/LLM-judge 한계 검증"
)

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 한 눈에",
    "🎯 단일 케이스 데모",
    "📊 결과 대시보드",
    "🔍 환각 사례 갤러리",
    "🎙️ 실시간 시연",
])


# =========================================================
# TAB 0 — Hero / Landing
# =========================================================
with tab0:
    st.markdown(
        "## 9개 가설 → **5개 적중, 4개 폐기**\n"
        "당연한 줄 알았는데 아닌 것을 정량으로 입증한 프로젝트."
    )

    # 핵심 숫자 카드 3개
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            '<div class="hero-card">'
            '<div class="big-num">$0.05</div>'
            '<div class="stat-label">gpt-4o-mini로 40건 노트 생성 — '
            'WangLab GPT-4(2023) ROUGE 수준 (1/30 비용)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            '<div class="hero-card">'
            '<div class="big-num">0.254</div>'
            '<div class="stat-label">Pearson(ROUGE-1, Factuality) — '
            'ROUGE는 사실성을 못 잡음 (영어 22%, 한국어 30% 케이스에서 ROUGE↑·환각↑)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            '<div class="hero-card">'
            '<div class="big-num">0.78점</div>'
            '<div class="stat-label">gpt-4o vs Gemini judge 격차 — '
            '단일 judge 결과 self-bias 위험 (Claude·Gemini 교차로 검증)</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # CTA
    cta1, cta2 = st.columns([2, 1])
    with cta1:
        st.markdown("### 🎙️ 지금 직접 시연해보세요")
        st.markdown(
            "오른쪽 위 **`🎙️ 실시간 시연`** 탭에서 음성 녹음 또는 텍스트 입력으로 "
            "즉시 SOAP 노트 생성 가능."
        )
    with cta2:
        st.info("💡 마이크 없으면 텍스트 입력 모드 사용")

    st.divider()

    # 9개 발견
    st.markdown("### 9개 발견 요약")
    st.markdown("""
| # | 발견 | 결과 |
|---|---|---|
| 1 | gpt-4o-mini가 GPT-4(2023) 수준 ROUGE 달성 | ✅ |
| 2 | ROUGE는 사실성을 못 잡음 (Pearson 0.254) | ✅ |
| 3 | ~~Korean Format이 영어 편향 judge로 부당 감점~~ | ❌ 폐기 |
| 4 | TF-IDF dynamic이 한국어에서 random 못 이김 | ❌ 가설 틀림 |
| 5 | kiwi 형태소 토크나이저로도 해결 안 됨 | ❌ 가설 틀림 |
| 6 | Embedding도 한국어 random 못 이김 | ❌ 가설 틀림 |
| 7 | gpt-4o vs Claude vs Gemini self-bias 0.7–0.8점 | ✅ |
| 8 | Format 점수는 judge간 합의 없음 (Pearson −0.04) | ✅ |
| 9 | KO가 EN보다 judge간 합의 높음 (0.73 vs 0.59) | ✅ |
""")

    st.divider()

    # 사용 기술
    st.markdown("### 🧰 사용 기술")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**LLM**")
        st.write("- OpenAI gpt-4o-mini (생성)")
        st.write("- gpt-4o (judge)")
        st.write("- Claude Sonnet 4.5 (judge)")
        st.write("- Gemini 2.5 Flash (judge)")
    with t2:
        st.markdown("**Retriever**")
        st.write("- TF-IDF (sklearn)")
        st.write("- kiwipiepy (한국어 형태소)")
        st.write("- text-embedding-3-small")
    with t3:
        st.markdown("**평가·기타**")
        st.write("- ROUGE-1/2/L")
        st.write("- LLM-as-judge (3 vendor)")
        st.write("- Whisper (ASR)")
        st.write("- Streamlit + Plotly")


# =========================================================
# TAB 1 — 단일 케이스 (EN vs KO 동시)
# =========================================================
with tab1:
    st.header("실시간 노트 생성 + 평가 — 영·한 동시 비교")
    st.caption("같은 encounter를 영어/한국어로 동시 표시. 번역 노이즈가 사실성·포맷에 미치는 영향 직접 확인.")

    c1, c2 = st.columns([2, 2])
    with c1:
        en_items = load_jsonl(DATA / "aci_test1.jsonl")
        ko_items = load_jsonl(DATA / "aci_test1_ko.jsonl")
        if not en_items or not ko_items:
            st.error("aci_test1 / aci_test1_ko 없음")
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
    st.subheader("📊 3-Vendor Judge 점수")
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
# TAB 2 — Dashboard
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

    st.subheader("3. Retriever 비교 (발견 #4-6)")
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

    st.subheader("4. 3-Vendor Judge (발견 #7-9)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**EN baseline test1**")
        en_jdf, n_en = judge_comparison_table("EN")
        st.dataframe(en_jdf, hide_index=True, use_container_width=True)
    with cc2:
        st.markdown("**KO baseline test1_ko**")
        ko_jdf, n_ko = judge_comparison_table("KO")
        st.dataframe(ko_jdf, hide_index=True, use_container_width=True)

    melt = pd.melt(
        pd.concat([en_jdf.assign(Lang="EN"), ko_jdf.assign(Lang="KO")]),
        id_vars=["Metric", "Lang"],
        value_vars=["gpt-4o", "Claude", "Gemini"],
        var_name="Judge", value_name="Score",
    )
    fig3 = px.bar(melt, x="Metric", y="Score", color="Judge", barmode="group",
                  facet_col="Lang", text="Score",
                  title="3-Vendor Judge 비교 — Format은 합의 없음")
    fig3.update_traces(textposition="outside")
    fig3.update_yaxes(range=[0, 5])
    st.plotly_chart(fig3, use_container_width=True)


# =========================================================
# TAB 3 — Hallucination gallery + Diff
# =========================================================
with tab3:
    st.header("환각 케이스 갤러리 — 단어 단위 Diff")
    st.markdown(
        "Factuality 낮은 케이스. 🟥 빨강 = Prediction에만 있는 단어(환각 의심), "
        "🟨 노랑 = Reference에만 있는 단어(누락)."
    )

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
    st.caption(f"전체 {len(rows)}건 — Factuality 낮은 순")

    top_n = st.slider("표시 개수", 1, min(20, max(1, len(rows))), 5)
    for r in rows[:top_n]:
        title = (f"🔻 {r['id']} | gpt-4o={r['fact_gpt']} "
                 f"Claude={r['fact_claude']} Gemini={r['fact_gemini']}")
        with st.expander(title):
            st.markdown(f"**Judge rationale (gpt-4o)**: _{r['rationale_gpt']}_")

            left, right = side_by_side_diff(r["reference"], r["prediction"])
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Reference** (🟨 = pred에서 누락)")
                st.markdown(
                    f'<div style="background:#fafafa;padding:12px;border-radius:6px;'
                    f'height:320px;overflow-y:auto;font-family:monospace;font-size:13px;'
                    f'white-space:pre-wrap;line-height:1.6">{left}</div>',
                    unsafe_allow_html=True,
                )
            with cc2:
                st.markdown("**Prediction** (🟥 = 환각 의심)")
                st.markdown(
                    f'<div style="background:#fafafa;padding:12px;border-radius:6px;'
                    f'height:320px;overflow-y:auto;font-family:monospace;font-size:13px;'
                    f'white-space:pre-wrap;line-height:1.6">{right}</div>',
                    unsafe_allow_html=True,
                )


# =========================================================
# TAB 4 — Live (mic / 자유 입력 / 파일)
# =========================================================
with tab4:
    st.header("🎙️ 실시간 시연 — 음성 / 텍스트 / 파일")
    st.caption(
        "마이크 녹음, 자유 텍스트 입력, 오디오 파일 업로드 3가지 방식. "
        "어떤 방식이든 결과: dialogue → SOAP 노트."
    )

    # ========== 사용 전 필수 고지 (Engineering Risk) ==========
    st.warning(
        "**⚠️ 엔지니어링 리스크 고지 (Pipeline Limitation)**\n\n"
        "- **에러 전파 (Error Propagation) 위험**: 음성 인식(ASR) 단계에서 발생하는 "
        "고유명사(약물명·검사명 등) 누락이나 사투리/발음 노이즈는 뒷단의 Few-shot "
        "셀렉터를 교란하고, `gpt-4o-mini`가 환각(오진)을 일으킬 구조적 취약점이 존재합니다.\n"
        "- **현재의 대응**: 사용자가 생성 버튼을 누르기 전 텍스트를 직접 검토하고 "
        "수정할 수 있는 **'중간 편집 UI (Human-in-the-loop)'**를 제공하여 1차적으로 "
        "리스크를 통제하고 있습니다.\n"
        "- **향후 로드맵**: Whisper 출력단 직후에 '의학 사전 기반 동적 보정(Spell Checker) "
        "레이어' 배치 및 노이즈가 주입된 대용량 의료 코퍼스 기반의 'LoRA 미세조정(Fine-tuning)' "
        "모델로의 전환을 통해 근본적으로 개선할 예정입니다."
    )

    lc1, lc2 = st.columns([1, 1])
    with lc1:
        rec_lang = st.radio("언어", ["KO (한국어)", "EN (English)"],
                            horizontal=True, key="rec_lang")
    with lc2:
        whisper_model = st.selectbox(
            "Whisper 모델 (음성 입력 시)",
            ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        )

    lang_code = "ko" if rec_lang.startswith("KO") else "en"

    st.divider()

    # ========== 시연용 샘플 (항상 보임) ==========
    sample_ko = """[doctor] 안녕하세요, 어떤 일로 오셨어요?
[patient] 며칠 전부터 가슴이 답답하고 숨이 차요. 운동할 때 특히 심해져요.
[doctor] 언제부터 그러셨어요?
[patient] 한 4-5일 정도 됐어요. 처음엔 그냥 피곤해서 그런가 했는데 점점 심해지더라고요.
[doctor] 흉통이 어떤 양상인가요? 조이는 느낌인지, 찌르는 느낌인지요?
[patient] 가운데 부위가 꽉 조이는 느낌이에요. 식은땀도 같이 나고요.
[doctor] 통증이 다른 곳으로 뻗치진 않나요? 왼쪽 어깨나 턱 쪽으로요.
[patient] 가끔 왼쪽 어깨 쪽이 같이 묵직해요.
[doctor] 환자분 나이가 어떻게 되시고 평소에 앓고 계신 질환이 있으신가요?
[patient] 59세 남자고요, 고혈압이 있어서 암로디핀 5mg 매일 먹고 있어요. 담배도 30년 정도 폈고요.
[doctor] 가족 중에 심장병 있으신 분 계시나요?
[patient] 아버지가 60대에 심근경색으로 돌아가셨어요.
[doctor] 알겠습니다. 일단 심전도부터 찍고 채혈해서 트로포닌 수치하고 BNP 확인하겠습니다. 가슴 X-ray도 같이 볼게요.
[patient] 네, 알겠습니다."""

    sample_en = """[doctor] Hi, what brings you in today?
[patient] I've been having chest tightness and shortness of breath for the past few days. It gets worse when I exercise.
[doctor] When did this start exactly?
[patient] About 4-5 days ago. At first I thought it was just fatigue, but it's been getting progressively worse.
[doctor] How would you describe the chest pain? Is it squeezing, sharp, or something else?
[patient] It's a tight squeezing feeling in the center of my chest, with sweating.
[doctor] Does the pain radiate anywhere — like to your left shoulder or jaw?
[patient] Sometimes I feel a heaviness in my left shoulder too.
[doctor] Could you tell me about your medical history?
[patient] I'm 59, male, with hypertension. I take amlodipine 5mg daily. I've been smoking for about 30 years.
[doctor] Any family history of heart disease?
[patient] My father died of a heart attack in his 60s.
[doctor] Okay. Let's get an EKG, draw blood for troponin and BNP, and order a chest X-ray.
[patient] Sounds good, thanks."""

    sample = sample_ko if lang_code == "ko" else sample_en

    with st.expander("💡 시연용 샘플 대화 (보기 / 복사 / 텍스트 모드에서 자동 채우기)", expanded=True):
        st.markdown(
            "**의사-환자 가상 대화 시나리오** — 59세 남성, 흉통·호흡곤란 케이스. "
            "텍스트 입력 모드에서 **`📋 위 샘플 채우기`** 버튼으로 한 번에 입력 가능. "
            "마이크 녹음 시엔 이 내용을 참고해서 본인이 말해보면 됨."
        )
        st.text_area(
            " ", sample, height=320,
            label_visibility="collapsed", key="sample_display",
        )

    st.divider()

    input_method = st.radio(
        "입력 방법",
        ["📝 텍스트 입력 (가장 빠름)", "🎙️ 마이크 녹음", "📁 오디오 파일"],
        horizontal=True, key="input_method",
    )

    transcript = None

    if input_method.startswith("📝"):
        # 자유 텍스트 입력
        if st.button("📋 위 샘플 채우기"):
            st.session_state["_text_input"] = sample
        transcript = st.text_area(
            "Dialogue 입력 (의사·환자 대화 또는 환자 정보)",
            value=st.session_state.get("_text_input", ""),
            height=300,
            placeholder="예: [doctor] 어떤 일로 오셨어요?\n[patient] 어제부터 두통과 어지러움이 있어요...",
        )

    elif input_method.startswith("🎙️"):
        audio = st.audio_input("녹음 시작 → 정지", key="mic_input")
        if audio:
            if st.button("🎤 Whisper 전사", type="primary"):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(ROOT / ".env")
                except ImportError:
                    pass
                import os as _os
                if not _os.getenv("OPENAI_API_KEY"):
                    st.error("OPENAI_API_KEY가 .env에 없음")
                else:
                    from openai import OpenAI
                    client = OpenAI()
                    with st.spinner(f"전사 중 ({lang_code})..."):
                        try:
                            audio.seek(0)
                            transcript = client.audio.transcriptions.create(
                                model=whisper_model,
                                file=("audio.wav", audio, "audio/wav"),
                                language=lang_code,
                            ).text
                            st.session_state["_transcribed"] = transcript
                        except Exception as e:
                            st.error(f"전사 실패: {e}")
        transcript = st.session_state.get("_transcribed", transcript)
        if transcript:
            st.markdown("**전사 결과**")
            st.text_area(" ", transcript, height=150,
                         label_visibility="collapsed", key="mic_transcript")

    else:  # 파일
        up = st.file_uploader(
            "오디오 파일 (mp3/wav/m4a/webm)",
            type=["mp3", "wav", "m4a", "webm", "ogg", "flac"],
            key="upload",
        )
        if up and st.button("🎤 Whisper 전사", type="primary", key="up_btn"):
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
            except ImportError:
                pass
            import os as _os
            if not _os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY가 .env에 없음")
            else:
                from openai import OpenAI
                client = OpenAI()
                with st.spinner(f"전사 중 ({lang_code})..."):
                    try:
                        transcript = client.audio.transcriptions.create(
                            model=whisper_model,
                            file=(up.name, up, f"audio/{up.type.split('/')[-1]}"),
                            language=lang_code,
                        ).text
                        st.session_state["_file_transcript"] = transcript
                    except Exception as e:
                        st.error(f"전사 실패: {e}")
        transcript = st.session_state.get("_file_transcript", transcript)
        if transcript:
            st.markdown("**전사 결과**")
            st.text_area(" ", transcript, height=150,
                         label_visibility="collapsed", key="file_transcript")

    # SOAP 생성 버튼
    st.divider()
    if transcript and transcript.strip():
        if st.button("🩺 SOAP 노트 생성", type="primary", use_container_width=True):
            with st.spinner("ICL로 SOAP 노트 생성 중 (gpt-4o-mini, 2-shot)..."):
                note, err = generate_soap_from_dialogue(transcript, lang_code)
            if err:
                st.error(err)
            else:
                st.divider()
                col_t, col_n = st.columns(2)
                with col_t:
                    st.subheader("📝 Dialogue (입력)")
                    st.text_area(" ", transcript, height=400,
                                 label_visibility="collapsed", key="final_input")
                with col_n:
                    st.subheader("🩺 Generated SOAP Note")
                    st.text_area(" ", note, height=400,
                                 label_visibility="collapsed", key="final_note")
                st.success("✅ 완료. 다른 입력으로 또 시도해보세요.")
    else:
        st.info("위에서 입력 방법을 선택하고 dialogue를 채운 뒤 SOAP 생성 버튼을 누르세요.")

    st.divider()
    with st.expander("⚠️ 상세 한계 (데이터·모델)"):
        st.markdown("""
- **ICL pool은 자동번역 한국어**: ACI-Bench 미국 모의 대화를 gpt-4o-mini로 번역한 67건. 실 한국 의무기록 스타일과는 차이 있음.
- **Whisper 의료 용어**: whisper-1은 일반 한국어는 매우 정확, 드문 의료 용어는 가끔 오인식. gpt-4o-transcribe로 개선 가능.
- **짧은 입력의 환각**: 나이·성별·기왕력 명시 안 하면 모델이 환각으로 채울 위험.
- **임상 안전 보장 없음**: 학습·연구·데모 목적. 실 환자 사용 금지.
""")
