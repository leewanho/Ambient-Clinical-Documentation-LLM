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

import html
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
st.markdown(
    """
    <style>
    /* Document Card Viewers */
    .doc-viewer-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .doc-viewer-card:hover {
        box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.05);
        border-color: #cbd5e1;
    }
    
    .doc-viewer-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
    }
    
    .doc-viewer-dlg::before {
        background: #8b5cf6 !important;
    }
    .doc-viewer-ref::before {
        background: #10b981 !important;
    }
    .doc-viewer-pred::before {
        background: #3b82f6 !important;
    }
    
    .doc-viewer-header {
        padding: 0.8rem 1.2rem;
        border-bottom: 1px solid #f1f5f9;
        background: #f8fafc;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .doc-viewer-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .doc-viewer-body {
        padding: 1.2rem;
        overflow-y: auto;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #334155;
        white-space: pre-wrap;
    }
    
    .doc-viewer-body::-webkit-scrollbar {
        width: 6px;
    }
    .doc-viewer-body::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    .doc-viewer-body::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 9999px;
    }
    .doc-viewer-body::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    .filter-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Custom Dashboard Tables */
    .custom-dashboard-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        font-family: inherit;
        background-color: #ffffff;
    }
    
    .custom-dashboard-table th {
        background-color: #f8fafc;
        color: #1e293b;
        font-weight: 700;
        padding: 0.75rem 1rem;
        border-bottom: 2px solid #e2e8f0;
        font-size: 0.95rem;
        text-align: left;
    }
    
    .custom-dashboard-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.95rem;
        color: #334155;
    }
    
    .custom-dashboard-table tr:last-child td {
        border-bottom: none;
    }
    
    .custom-dashboard-table tr:hover td {
        background-color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

    with st.container(border=True):
        st.markdown('<div class="filter-card-title">🔍 대화 분석 및 검색 조건 설정</div>', unsafe_allow_html=True)
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
            
            # 1. Dialogue
            dlg_content = html.escape(case["messages"][1]["content"])
            st.markdown(f"""
            <div class="doc-viewer-card doc-viewer-dlg">
                <div class="doc-viewer-header">
                    <span class="doc-viewer-title">💬 Dialogue (대화 입력)</span>
                </div>
                <div class="doc-viewer-body" style="height: 250px;">{dlg_content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Reference
            ref_content = html.escape(case["messages"][2]["content"])
            st.markdown(f"""
            <div class="doc-viewer-card doc-viewer-ref">
                <div class="doc-viewer-header">
                    <span class="doc-viewer-title">🎯 Reference (임상 정답)</span>
                </div>
                <div class="doc-viewer-body" style="height: 200px;">{ref_content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Generated
            if pred and pred.get("prediction"):
                pred_content = html.escape(pred["prediction"])
                st.markdown(f"""
                <div class="doc-viewer-card doc-viewer-pred">
                    <div class="doc-viewer-header">
                        <span class="doc-viewer-title">🤖 Generated (모델 출력)</span>
                    </div>
                    <div class="doc-viewer-body" style="height: 200px;">{pred_content}</div>
                </div>
                """, unsafe_allow_html=True)
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
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# =========================================================
# TAB 2 — dashboard
# =========================================================
with tab2:
    st.header("결과 대시보드")
    st.markdown('<p class="tab-subtitle">실험 종합 통계, ROUGE의 한계, 리트리버 성능 비교, 판사 간 Self-bias 정량 검증.</p>', unsafe_allow_html=True)

    # Helper function to style Plotly figures beautifully (Indigo & Slate theme)
    def style_plotly_fig(fig):
        fig.update_layout(
            font_family="Outfit, Noto Sans KR, sans-serif",
            title_font=dict(size=16, color="#0f172a", family="Outfit, Noto Sans KR"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=12, color="#475569")
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=12, color="#475569")
        )
        return fig

    # Helper to clean leading whitespace for markdown rendering safety
    def clean_html(html_str):
        return "\n".join(line.strip() for line in html_str.split("\n"))

    # Helper to convert main metrics dataframe to premium HTML
    def df_to_html_table(df):
        html_str = """
        <div style="overflow-x:auto;">
            <table class="custom-dashboard-table">
                <thead>
                    <tr>
                        <th>Language</th>
                        <th>Retriever</th>
                        <th style="text-align: center;">N</th>
                        <th style="text-align: center;">ROUGE-1</th>
                        <th style="text-align: center;">ROUGE-L</th>
                        <th style="text-align: center;">Factuality</th>
                        <th style="text-align: center;">Completeness</th>
                        <th style="text-align: center;">Format</th>
                    </tr>
                </thead>
                <tbody>
        """
        for _, row in df.iterrows():
            lang = row['Language']
            flag = "🇺🇸 EN" if lang == "EN" else "🇰🇷 KO"
            retriever = row['Retriever']
            n = row['N']
            r1 = f"{row['ROUGE-1']:.4f}" if pd.notna(row['ROUGE-1']) else "-"
            rl = f"{row['ROUGE-L']:.4f}" if pd.notna(row['ROUGE-L']) else "-"
            fact = f"{row['Factuality']:.2f}" if pd.notna(row['Factuality']) else "-"
            comp = f"{row['Completeness']:.2f}" if pd.notna(row['Completeness']) else "-"
            fmt = f"{row['Format']:.2f}" if pd.notna(row['Format']) else "-"
            
            # Highlight best-in-class row
            is_highlight = False
            if lang == "EN" and retriever == "TF-IDF":
                is_highlight = True
            elif lang == "KO" and retriever == "Random":
                is_highlight = True
                
            row_style = "background-color: #faf5ff; font-weight: 600;" if is_highlight else ""
            
            html_str += f"""
                    <tr style="{row_style}">
                        <td><strong>{flag}</strong></td>
                        <td><code>{retriever}</code></td>
                        <td style="text-align: center;">{n}</td>
                        <td style="text-align: center;">{r1}</td>
                        <td style="text-align: center;">{rl}</td>
                        <td style="text-align: center; color: #166534; font-weight: 700;">{fact}</td>
                        <td style="text-align: center;">{comp}</td>
                        <td style="text-align: center;">{fmt}</td>
                    </tr>
            """
        html_str += """
                </tbody>
            </table>
        </div>
        """
        return clean_html(html_str)

    # Helper to convert judge metrics dataframe to HTML
    def judge_df_to_html_table(df):
        html_str = """
        <table class="custom-dashboard-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th style="text-align: center;">gpt-4o</th>
                    <th style="text-align: center;">Claude</th>
                    <th style="text-align: center;">Gemini</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in df.iterrows():
            metric = row['Metric'].capitalize()
            g = f"{row['gpt-4o']:.2f}" if pd.notna(row['gpt-4o']) else "-"
            c = f"{row['Claude']:.2f}" if pd.notna(row['Claude']) else "-"
            m = f"{row['Gemini']:.2f}" if pd.notna(row['Gemini']) else "-"
            
            # Highlight self-bias vendor (gpt-4o) column
            html_str += f"""
                <tr>
                    <td><strong>{metric}</strong></td>
                    <td style="text-align: center; font-weight: 700; color: #4f46e5; background-color: #f5f3ff;">{g}</td>
                    <td style="text-align: center;">{c}</td>
                    <td style="text-align: center;">{m}</td>
                </tr>
            """
        html_str += """
            </tbody>
        </table>
        """
        return clean_html(html_str)

    # --- Section 1: Retriever × Language 종합 ---
    with st.container(border=True):
        st.markdown("### 📊 1. Retriever × Language 종합 성능")
        st.markdown('<p class="tab-subtitle" style="margin-top:-0.5rem !important; margin-bottom: 0.5rem !important;">각 언어 및 검색 조건에 따른 ROUGE 어휘 유사도와 LLM-judge 평가 점수의 전체 종합 결과입니다.</p>', unsafe_allow_html=True)
        df = compute_metrics_table()
        st.markdown(df_to_html_table(df), unsafe_allow_html=True)

    st.write("") # Spacer

    # --- Section 2: ROUGE vs Factuality ---
    with st.container(border=True):
        st.markdown("### 🎯 2. ROUGE는 사실성(Factuality)을 못 잡는다 (발견 #2)")
        st.markdown('<p class="tab-subtitle" style="margin-top:-0.5rem !important; margin-bottom: 1rem !important;">전통적인 n-gram 중첩 메트릭(ROUGE)과 최신 LLM Judge가 매긴 Factuality 간의 Pearson 상관계수를 시각화합니다. (상관이 매우 낮아 ROUGE 지표 맹신은 위험함)</p>', unsafe_allow_html=True)
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
                             title=f"Pearson r = {r:.3f} (ROUGE와 사실성의 불일치)")
            fig.update_traces(
                marker=dict(size=12, opacity=0.8, color="#4f46e5", line=dict(width=1, color="white")),
                hovertemplate="<b>Case ID:</b> %{customdata[0]}<br><b>ROUGE-1:</b> %{x:.4f}<br><b>Factuality:</b> %{y:.2f}"
            )
            style_plotly_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.write("") # Spacer

    # --- Section 3: Retriever 비교 ---
    with st.container(border=True):
        st.markdown("### 📈 3. Retriever 비교 — KO에서 Random이 최강 (발견 #4-6)")
        st.markdown('<p class="tab-subtitle" style="margin-top:-0.5rem !important; margin-bottom: 1rem !important;">리트리버 방식(Random vs TF-IDF vs Embedding)에 따른 성능을 사실성(Factuality)과 ROUGE 지표를 통해 크로스 비교합니다.</p>', unsafe_allow_html=True)
        sub_df = df.dropna(subset=["Factuality"])
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            fig = px.bar(sub_df, x="Retriever", y="Factuality", color="Language",
                         barmode="group", text="Factuality",
                         color_discrete_map={"EN": "#0ea5e9", "KO": "#8b5cf6"},
                         title="Factuality by Retriever × Language")
            fig.update_traces(
                textposition="outside", 
                texttemplate="%{text:.2f}",
                marker=dict(line=dict(width=0))
            )
            fig.update_yaxes(range=[0, 5])
            style_plotly_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            
        with c_chart2:
            fig2 = px.bar(sub_df, x="Retriever", y="ROUGE-1", color="Language",
                          barmode="group", text="ROUGE-1",
                          color_discrete_map={"EN": "#0ea5e9", "KO": "#8b5cf6"},
                          title="ROUGE-1 by Retriever × Language")
            fig2.update_traces(
                textposition="outside", 
                texttemplate="%{text:.4f}",
                marker=dict(line=dict(width=0))
            )
            style_plotly_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    st.write("") # Spacer

    # --- Section 4: 3-Vendor Judge ---
    with st.container(border=True):
        st.markdown("### ⚖️ 4. 3-Vendor Judge 및 자가 편향(Self-bias) 검증 (발견 #7-9)")
        st.markdown('<p class="tab-subtitle" style="margin-top:-0.5rem !important; margin-bottom: 1rem !important;">동일한 요약 결과에 대해 3개의 대형 LLM 판사 모델(GPT-4o, Claude, Gemini)이 매긴 점수 비교입니다. (자가 모델 출력인 GPT-4o-mini에 더 호의적인 Self-bias 0.7~0.8점 격차가 나타남)</p>', unsafe_allow_html=True)
        
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("<p style='font-weight:700; color:#1e293b; margin-bottom: 0.2rem;'>🇺🇸 EN baseline (test1)</p>", unsafe_allow_html=True)
            en_jdf, n_en = judge_comparison_table("EN")
            st.markdown(judge_df_to_html_table(en_jdf), unsafe_allow_html=True)
        with cc2:
            st.markdown("<p style='font-weight:700; color:#1e293b; margin-bottom: 0.2rem;'>🇰🇷 KO baseline (test1_ko)</p>", unsafe_allow_html=True)
            ko_jdf, n_ko = judge_comparison_table("KO")
            st.markdown(judge_df_to_html_table(ko_jdf), unsafe_allow_html=True)
            
        st.write("") # Inner spacing
        
        melt = pd.melt(
            pd.concat([en_jdf.assign(Lang="EN"), ko_jdf.assign(Lang="KO")]),
            id_vars=["Metric", "Lang"],
            value_vars=["gpt-4o", "Claude", "Gemini"],
            var_name="Judge", value_name="Score",
        )
        fig3 = px.bar(melt, x="Metric", y="Score", color="Judge", barmode="group",
                      facet_col="Lang", text="Score",
                      color_discrete_map={"gpt-4o": "#4f46e5", "Claude": "#ec4899", "Gemini": "#f59e0b"},
                      title="3-Vendor Judge 비교 — Format은 합의 없음")
        fig3.update_traces(
            textposition="outside", 
            texttemplate="%{text:.2f}",
            marker=dict(line=dict(width=0))
        )
        fig3.update_yaxes(range=[0, 5])
        style_plotly_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    st.write("") # Spacer

    # --- Section 5: 9개 발견 요약 ---
    with st.container(border=True):
        st.markdown("### 📌 5. 9개 발견 요약")
        st.markdown('<p class="tab-subtitle" style="margin-top:-0.5rem !important; margin-bottom: 0.5rem !important;">본 프로젝트에서 정량적으로 검증하고 규명한 9가지 가설 검증 결과의 최종 테이블 요약입니다.</p>', unsafe_allow_html=True)
        st.markdown("""
| # | 발견 | 유형 | 가설 결과 |
|---|---|---|---|
| 1 | gpt-4o-mini가 GPT-4(2023) ROUGE 동급 ($0.05/40건) | 사전 가설 | ✅ 가설 적중 |
| 2 | ROUGE는 사실성 못 잡음 (Pearson 0.254) | 사전 가설 | ✅ 가설 적중 |
| 3 | ~~한국어 Format 부당 감점~~ | 사전 가설 | ❌ 가설 폐기 |
| 4 | TF-IDF dynamic이 한국어에서 random 못 이김 | 사전 가설 | ❌ 가설 폐기 |
| 5 | kiwi 형태소 토크나이저로도 해결 안 됨 | 단계 검증 가설 | ❌ 가설 폐기 |
| 6 | Embedding도 KO에서 random 못 이김 | 단계 검증 가설 | ❌ 가설 폐기 |
| 7 | gpt-4o vs Claude vs Gemini self-bias 0.7-0.8점 | 외부 통용 검증 | ✅ 가설 적중 |
| 8 | Format 점수는 judge간 합의 없음 (Pearson −0.04) | 사후 패턴 발견 | ✅ 가설 적중 |
| 9 | KO가 EN보다 judge간 합의 높음 (0.73 vs 0.59) | 사후 패턴 발견 | ✅ 가설 적중 |
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

    # 두 개의 컬럼으로 분할 (좌측: 입력 인터페이스, 우측: 리스크 고지 경고창)
    input_col, warn_col = st.columns([1.2, 1.8])

    with input_col:
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

    with warn_col:
        # 🛑 상시 노출하는 '선제 방어막' (리스크 고지)
        st.warning("""
⚠️ **엔지니어링 리스크 고지 (Pipeline Limitation)**
* **에러 전파 (Error Propagation) 위험:** 음성 인식(ASR) 단계에서 발생하는 고유명사(약물명, 검사명 등) 누락이나 사투리/발음 노이즈는 뒷단의 Few-shot 셀렉터를 교란하고, `gpt-4o-mini`가 환각(오진)을 일으킬 구조적 취약점이 존재합니다.
* **현재의 대응:** 사용자가 생성 버튼을 누르기 전 텍스트를 직접 검토하고 수정할 수 있는 **'중간 편집 UI (Human-in-the-loop)'**를 제공하여 1차적으로 리스크를 통제하고 있습니다.
* **향후 로드맵:** Whisper 출력단 직후에 '의학 사전 기반 동적 보정(Spell Checker) 레이어' 배치 및 노이즈가 주입된 대용량 의료 코퍼스 기반의 'LoRA 미세조정(Fine-tuning)' 모델로의 전환을 통해 근본적으로 개선할 예정입니다.
""")

    # Initialize session state keys if not present
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "generated_note" not in st.session_state:
        st.session_state.generated_note = ""

    # 생성 버튼
    if audio_data:
        if st.button("🎙️ 1단계: 음성 전사 실행 (Transcribe Audio)", type="secondary"):
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
                    st.session_state.transcript = client.audio.transcriptions.create(
                        model=whisper_model,
                        file=("audio.wav", audio_data, "audio/wav"),
                        language=lang_code,
                    ).text
                    st.session_state.generated_note = ""  # 새로운 전사 시 이전 노트 초기화
                except Exception as e:
                    st.error(f"전사 실패: {e}")
                    st.stop()

    if st.session_state.transcript:
        st.divider()
        st.subheader("📝 1단계: Whisper 변환 결과 (편집 가능)")
        
        # 변환된 텍스트를 의사가 직접 수정할 수 있는 text_area 제공
        edited_text = st.text_area(
            "인식된 대화 내용입니다. 오타나 의학 용어 오류가 있다면 수정 후 아래 생성 버튼을 누르세요.",
            value=st.session_state.transcript,
            height=250,
            key="edited_transcript_area"
        )

        if st.button("🩺 2단계: SOAP 진료 기록 생성", type="primary"):
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
            user_q = f"Conversation:\n{edited_text}\n\nGenerate Clinical Note:"
            msgs.append({"role": "user", "content": user_q})

            with st.spinner("📝 SOAP 노트 생성 중 (gpt-4o-mini, 2-shot)..."):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini", messages=msgs,
                        temperature=0.2, max_tokens=2048,
                    )
                    st.session_state.generated_note = resp.choices[0].message.content
                except Exception as e:
                    st.error(f"노트 생성 실패: {e}")
                    st.stop()

            st.success(
                f"완료! Few-shot 예시 ID: "
                f"{[e.get('meta',{}).get('encounter_id','?') for e in few]}"
            )

    if st.session_state.generated_note:
        st.divider()
        st.subheader("🩺 Generated SOAP Note")
        st.text_area(" ", st.session_state.generated_note, height=350,
                     label_visibility="collapsed", key="note_out")

    st.divider()
    with st.expander("⚠️ 한계 (Limitations)"):
        st.markdown("""
- **Cross-Border 도메인 격차**: 미국 ACI-Bench 기반 데이터셋이라 한국 의료 환경 특유의 보험 수가 체계, 의약품 처방 트렌드, 진료 차팅 관습을 반영하기 어렵습니다. 단순한 번역체 노이즈 문제를 넘어 본질적인 국가 간 의료 도메인의 차이(Gap)가 존재합니다.
- **Random 2-shot의 변동성(Variance)**: 한국어 환경에서 정교한 임베딩이나 TF-IDF 리트리버 기반 매칭이 모두 실패하여 대안으로 무작위(Random) 선택 방식을 채택했습니다. 그러나 이 방식은 상용화된 서비스 환경에서 사용자가 매번 생성 요청을 보낼 때마다 출력의 품질과 스타일 변동성을 통제하기 어렵다는 근본적 한계를 지닙니다.
- **메트릭 디커플링 및 Rubric 한계**: ROUGE 점수가 의학적 사실성(Factuality) 평가 지표와 상충하고 디커플링됨을 증명했음에도, 퓨샷 아키텍처의 필터링 및 선별 과정에 ROUGE 메트릭이 개입되어 있습니다. 또한 포맷 평가(Format)에서 채점 AI 판사들 간의 합의율이 음수(-0.04)를 기록하는 등, 일부 항목에서 평가 기준의 느슨함이 확인되었으며, 평가 데이터 수($n=40$)의 통계적 유의성 문제도 남아있습니다.
""")
