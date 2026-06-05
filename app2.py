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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    /* Global Font Override */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', 'Noto Sans KR', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .main-header h1 {
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        font-size: 2.8rem;
    }

    /* Metric Cards */
    .hero-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
        height: 100%;
    }

    .hero-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(to bottom, #3b82f6, #6366f1);
    }

    .hero-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 20px -5px rgba(0, 0, 0, 0.08), 0 6px 8px -6px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }

    .card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #4f46e5;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .big-num {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        line-height: 1.1;
        margin-bottom: 0.6rem;
        background: linear-gradient(135deg, #1e293b 30%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.5;
    }

    /* Technical Pipeline Flowchart */
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: white;
        padding: 2.2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .pipeline-step {
        flex: 1;
        text-align: center;
        padding: 0.25rem;
    }

    .step-icon {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }

    .step-name {
        font-weight: 700;
        color: #0f172a;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }

    .step-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
    }

    .pipeline-arrow {
        font-size: 1.8rem;
        color: #cbd5e1;
        padding: 0 0.5rem;
        font-weight: bold;
    }

    /* CTA Card (SaaS Launch Style) */
    .cta-card {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid #a5b4fc;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .cta-card:hover {
        box-shadow: 0 12px 20px -3px rgba(59, 130, 246, 0.1);
        transform: translateY(-2px);
    }

    .cta-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: #4f46e5;
        color: white;
        font-size: 0.68rem;
        font-weight: 800;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        letter-spacing: 0.5px;
    }

    .cta-card h3 {
        color: #312e81;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 0.6rem;
        font-size: 1.2rem;
    }

    .cta-card p {
        color: #3730a3;
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Tech Stack Tag Grid */
    .tech-stack-grid {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }

    .tech-category {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }

    .category-name {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .tag-group {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }

    .tech-tag {
        font-size: 0.78rem;
        font-weight: 500;
        color: #0f172a;
        background: #f1f5f9;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }

    .tech-tag:hover {
        background: #e2e8f0;
        transform: translateY(-1px);
    }

    /* Findings 3x3 Grid */
    .findings-grid-3x3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }

    @media (max-width: 1024px) {
        .findings-grid-3x3 {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 768px) {
        .findings-grid-3x3 {
            grid-template-columns: 1fr;
        }
    }

    .finding-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: all 0.25s ease;
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }

    .finding-card:hover {
        box-shadow: 0 10px 18px rgba(0,0,0,0.06);
        border-color: #cbd5e1;
        transform: translateY(-4px);
    }

    .finding-card-top {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .finding-num {
        background: #f1f5f9;
        color: #475569;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.25rem 0.55rem;
        border-radius: 6px;
        min-width: 32px;
        text-align: center;
    }

    .finding-title {
        font-weight: 700;
        color: #1e293b;
        font-size: 0.92rem;
        line-height: 1.3;
        flex: 1;
    }

    .finding-badge {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.5rem;
        border-radius: 9999px;
        white-space: nowrap;
        align-self: flex-start;
    }

    .badge-hit {
        background-color: #dcfce7;
        color: #166534;
    }

    .badge-discard {
        background-color: #fee2e2;
        color: #991b1b;
    }

    .finding-detail {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.5;
    }

    /* Table Styling */
    div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }

    div[data-testid="stMarkdownContainer"] th {
        background-color: #f8fafc;
        color: #1e293b;
        font-weight: 600;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 2px solid #e2e8f0;
        font-size: 0.9rem;
    }

    div[data-testid="stMarkdownContainer"] td {
        padding: 12px 16px;
        border-bottom: 1px solid #f1f5f9;
        color: #334155;
        background-color: white;
        font-size: 0.9rem;
    }

    div[data-testid="stMarkdownContainer"] tr:last-child td {
        border-bottom: none;
    }

    div[data-testid="stMarkdownContainer"] tr:hover td {
        background-color: #f8fafc;
    }

    /* Tabs UI */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-size: 0.92rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
        padding: 0 20px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.6);
        color: #0f172a;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>📋 Ambient Clinical Documentation LLM</h1>
        <p style="margin-top:0.8rem; font-size:1.4rem; opacity:0.85; line-height:1.5; font-weight: 400;">
            MEDIQA-Chat 2023 (ACI-Bench Task B) 재현 · 한국어 확장<br>
            ROUGE / LLM-judge 한계 검증 · 9개 가설 정량 테스트
        </p>
    </div>
    """,
    unsafe_allow_html=True,
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
    st.markdown("## 프로젝트 한 눈에")
    st.caption("풀스택 임상 노트 파이프라인 구축 + 9개 가설 정량 검증.")

    # 상단 2열 배치 (40% : 60%)
    col_left, col_right = st.columns([1.6, 2.4])

    with col_left:
        # CTA (Launch Style) 카드
        st.markdown(
            """
            <div class="cta-card">
                <div class="cta-badge">LIVE DEMO</div>
                <h3>🎙️ 지금 직접 시연해보세요</h3>
                <p>상단 <b>🎙️ 실시간 시연</b> 탭에서 대화 음성 녹음 또는 자유 텍스트 입력으로 즉시 SOAP 임상 노트를 생성하고 성능을 체험해 보세요.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("💡 마이크가 없으신 경우 텍스트 입력 모드를 활용해 테스트하실 수 있습니다.")

        # 사용 기술 배지 그리드
        st.markdown(
            """
            <div class="tech-stack-grid">
                <div class="tech-category">
                    <div class="category-name">🤖 LLM & Generation</div>
                    <div class="tag-group">
                        <span class="tech-tag">gpt-4o-mini (생성)</span>
                        <span class="tech-tag">gpt-4o (평가)</span>
                        <span class="tech-tag">Claude 4.5 (평가)</span>
                        <span class="tech-tag">Gemini 2.5 (평가)</span>
                    </div>
                </div>
                <div class="tech-category" style="margin-top: 0.5rem;">
                    <div class="category-name">🔍 Retrieval Engine</div>
                    <div class="tag-group">
                        <span class="tech-tag">TF-IDF (sklearn)</span>
                        <span class="tech-tag">kiwipiepy (형태소)</span>
                        <span class="tech-tag">text-embedding-3</span>
                    </div>
                </div>
                <div class="tech-category" style="margin-top: 0.5rem;">
                    <div class="category-name">⚖️ Evaluation & ASR</div>
                    <div class="tag-group">
                        <span class="tech-tag">Whisper ASR (음성)</span>
                        <span class="tech-tag">ROUGE-1/L</span>
                        <span class="tech-tag">Pearson 상관계수</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        # 핵심 카드 3개 (가로 정렬)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                '<div class="hero-card">'
                '<div class="card-title">[ 빌드 규모 ]</div>'
                '<div class="big-num">7×3×2</div>'
                '<div class="stat-label">7 retriever × 3 vendor judge × 2 lang(EN/KO) 전수 평가 파이프라인 구축 및 데모 구현</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                '<div class="hero-card">'
                '<div class="card-title">[ 메트릭 패러독스 ]</div>'
                '<div class="big-num">0.254</div>'
                '<div class="stat-label">Pearson(ROUGE-1, Factuality) — ROUGE는 사실성 평가 불가. 영 22%·한 30% 케이스에서 괴리 관측</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                '<div class="hero-card">'
                '<div class="card-title">[ 평가자 패러독스 ]</div>'
                '<div class="big-num">0.78점</div>'
                '<div class="stat-label">gpt-4o vs Gemini judge 격차 — 단일 judge 신뢰 불가. 교차 평가를 통한 self-bias 편향 입증</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # 시스템 파이프라인 가로 다이어그램
        st.markdown(
            """
            <div class="pipeline-flow">
                <div class="pipeline-step">
                    <div class="step-icon">🎙️</div>
                    <div class="step-name">1. ASR 전사</div>
                    <div class="step-desc">Whisper API 기반<br>의사 대화 음성 전사</div>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <div class="step-icon">🔍</div>
                    <div class="step-name">2. Few-shot 검색</div>
                    <div class="step-desc">유사도 검색 기반<br>최적의 진료 예시 선별</div>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <div class="step-icon">🤖</div>
                    <div class="step-name">3. SOAP 생성</div>
                    <div class="step-desc">gpt-4o-mini ICL<br>진료 기록 요약</div>
                </div>
                <div class="pipeline-arrow">➔</div>
                <div class="pipeline-step">
                    <div class="step-icon">⚖️</div>
                    <div class="step-name">4. 교차 검증</div>
                    <div class="step-desc">3-Vendor LLM Judge<br>사실성/형식 교차 검증</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # 하단 3x3 격자 카드 배치
    st.markdown("### 📊 9개 핵심 발견 및 연구 성과")
    st.caption("당연한 줄 알았는데 아닌 것을 정량으로 입증한 프로젝트.")

    st.markdown(
        """
        <div class="findings-grid-3x3">
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">01</div>
                    <div class="finding-title">gpt-4o-mini가 GPT-4 수준 달성</div>
                    <span class="finding-badge badge-hit">✅ 가설 적중</span>
                </div>
                <div class="finding-detail">단돈 0.05달러(약 70원)의 비용으로 과거 GPT-4 모델 성능 재현 (비용 1/30로 감축).</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">02</div>
                    <div class="finding-title">ROUGE는 임상 사실성 평가 불가</div>
                    <span class="finding-badge badge-hit">✅ 가설 적중</span>
                </div>
                <div class="finding-detail">상관관계 Pearson r = 0.254로 극히 저조. ROUGE 점수가 높아도 심각한 의학적 환각이 존재함을 입증.</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">03</div>
                    <div class="finding-title">Korean Format의 AI 판사 편향 감점 여부</div>
                    <span class="finding-badge badge-discard">❌ 가설 폐기</span>
                </div>
                <div class="finding-detail">주요 해외 LLM 판사들이 한국어 포맷팅이나 정렬 형태에 따른 편향 감점을 주지 않음을 정량 검증.</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">04</div>
                    <div class="finding-title">TF-IDF dynamic의 한국어 매칭 효과</div>
                    <span class="finding-badge badge-discard">❌ 가설 폐기</span>
                </div>
                <div class="finding-detail">한국어 환경에서는 무작위(Random)로 추출된 예시를 보여주는 것이 검색 기반 예시보다 더 우수함.</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">05</div>
                    <div class="finding-title">Kiwi 형태소 토크나이저의 성능 개선 여부</div>
                    <span class="finding-badge badge-discard">❌ 가설 폐기</span>
                </div>
                <div class="finding-detail">형태소 기반 토큰 분석을 활용해도 한국어 다이내믹 ICL 검색기의 성능 한계를 극복하지 못함.</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">06</div>
                    <div class="finding-title">임베딩(Embedding) 매칭의 한국어 성능</div>
                    <span class="finding-badge badge-discard">❌ 가설 폐기</span>
                </div>
                <div class="finding-detail">텍스트 임베딩 유사도 매칭 역시 무작위 선택 방식보다 낮은 임상 차팅 정확도를 기록함.</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">07</div>
                    <div class="finding-title">LLM Judge의 자가 편향(Self-bias) 규명</div>
                    <span class="finding-badge badge-hit">✅ 가설 적중</span>
                </div>
                <div class="finding-detail">gpt-4o vs Claude vs Gemini 채점 결과, 자기 가문 모델에 최대 0.7~0.8점의 편향 편차 규명.</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">08</div>
                    <div class="finding-title">Format 평가지표의 일관성 부재</div>
                    <span class="finding-badge badge-hit">✅ 가설 적중</span>
                </div>
                <div class="finding-detail">의무기록 포맷 평가 시 판사 모델 간 상관관계가 Pearson r = -0.04로 일치도 제로에 가깝음.</div>
            </div>
            <div class="finding-card">
                <div class="finding-card-top">
                    <div class="finding-num">09</div>
                    <div class="finding-title">국가/언어별 평가 일치도 격차</div>
                    <span class="finding-badge badge-hit">✅ 가설 적중</span>
                </div>
                <div class="finding-detail">한국어 평가의 판사 간 합의도(0.73)가 영어 평가의 합의도(0.59)보다 유의미하게 높게 관측됨.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
