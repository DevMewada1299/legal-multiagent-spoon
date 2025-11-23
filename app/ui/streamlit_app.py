import os
import sys
import tempfile
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.ingestion.pdf_ingestor import PDFIngestor
from graph_pipeline.graph import LegalAnalysisGraph

st.set_page_config(page_title="Legal Analyzer", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
    .badge {display:inline-block;padding:4px 8px;border-radius:12px;font-size:12px;margin-right:6px}
    .high {background:#ff4d4f;color:white}
    .medium {background:#faad14;color:white}
    .low {background:#1890ff;color:white}
    .pill {background:#2f54eb;color:white}
    .card {padding:16px;border-radius:12px;border:1px solid #e8e8e8;background:#ffffff; margin-bottom: 10px; color: #31333F;}
    .muted {color:#595959}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Legal Document Risk & Obligation Analyzer")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    model_choice = st.selectbox(
        "Model",
        options=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro"],
        index=0,
    )
    st.caption("The app will not store your key; it is used only for this session.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
analyze = st.button("Analyze", type="primary")

def severity_rank(val):
    order = {"high": 3, "medium": 2, "low": 1}
    return order.get((val or "").lower(), 0)

if analyze:
    if not api_key:
        st.error("Please enter your Gemini API key")
    elif not uploaded_file:
        st.error("Please upload a PDF document")
    else:
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GEMINI_MODEL_ANALYSIS"] = model_choice
        os.environ["GEMINI_MODEL_SUMMARY"] = model_choice
        os.environ["GEMINI_FALLBACK_ON_429"] = "true"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        with st.spinner("Analyzing document…"):
            ingestor = PDFIngestor()
            items = ingestor.ingest(tmp.name)
            graph = LegalAnalysisGraph()
            import asyncio
            final_state = asyncio.run(graph.run(clause_items=items))
        os.unlink(tmp.name)
        summary = final_state.get("summary", "")
        analysis = final_state.get("analysis", [])
        st.subheader("Analysis Results")
        summary_tab, risks_tab, obligations_tab, explorer_tab = st.tabs(["Executive Summary", "Key Risks", "Obligations Register", "Clause Explorer"])

        with summary_tab:
            st.markdown(f"<div class='card'>{summary}</div>", unsafe_allow_html=True)

        risk_rows = []
        obligation_rows = []
        for item in analysis:
            page = item.get("page")
            idx = item.get("index")
            for r in item.get("risks") or []:
                risk_rows.append({
                    "severity": (r.get("severity") or "").lower(),
                    "description": r.get("description"),
                    "category": r.get("category"),
                    "page": page,
                    "clause": idx,
                })
            for o in item.get("obligations") or []:
                obligation_rows.append({
                    "actor": o.get("actor"),
                    "action": o.get("action"),
                    "deadline": o.get("deadline"),
                    "page": page,
                    "clause": idx,
                })
        risk_rows.sort(key=lambda r: (-severity_rank(r["severity"]), r["page"] or 0))
        obligation_rows.sort(key=lambda o: (o["page"] or 0))
        with risks_tab:
            cols = st.columns(2)
            with cols[0]:
                for r in risk_rows[:5]:
                    sev = r["severity"] or "low"
                    sev_cls = "high" if sev == "high" else "medium" if sev == "medium" else "low"
                    badge = f"<span class='badge {sev_cls}'>{sev.capitalize()}</span>"
                    st.markdown(
                        f"<div class='card'>{badge} {r['description']}<div class='muted'>Page {r['page']} • Clause {r['clause']} • {r['category']}</div></div>",
                        unsafe_allow_html=True,
                    )
            with cols[1]:
                if len(risk_rows) > 5:
                    for r in risk_rows[5:10]:
                        sev = r["severity"] or "low"
                        sev_cls = "high" if sev == "high" else "medium" if sev == "medium" else "low"
                        badge = f"<span class='badge {sev_cls}'>{sev.capitalize()}</span>"
                        st.markdown(
                            f"<div class='card'>{badge} {r['description']}<div class='muted'>Page {r['page']} • Clause {r['clause']} • {r['category']}</div></div>",
                            unsafe_allow_html=True,
                        )

        with obligations_tab:
            cols2 = st.columns(2)
            with cols2[0]:
                for o in obligation_rows[:5]:
                    st.markdown(
                        f"<div class='card'><span class='badge pill'>Obligation</span> {o['actor'] or 'Actor'} – {o['action']}<div class='muted'>Page {o['page']} • Clause {o['clause']} • Deadline: {o['deadline'] or 'N/A'}</div></div>",
                        unsafe_allow_html=True,
                    )
            with cols2[1]:
                if len(obligation_rows) > 5:
                    for o in obligation_rows[5:10]:
                        st.markdown(
                            f"<div class='card'><span class='badge pill'>Obligation</span> {o['actor'] or 'Actor'} – {o['action']}<div class='muted'>Page {o['page']} • Clause {o['clause']} • Deadline: {o['deadline'] or 'N/A'}</div></div>",
                            unsafe_allow_html=True,
                        )

        with explorer_tab:
            for item in analysis:
                page = item.get("page")
                idx = item.get("index")
                excerpt = item.get("clause_excerpt")
                with st.expander(f"Clause {idx} (Page {page})"):
                    st.write(excerpt)
                    rs = item.get("risks") or []
                    os_ = item.get("obligations") or []
                    if rs:
                        for r in rs:
                            sev = (r.get("severity") or "low").lower()
                            sev_cls = "high" if sev == "high" else "medium" if sev == "medium" else "low"
                            badge = f"<span class='badge {sev_cls}'>{sev.capitalize()}</span>"
                            st.markdown(
                                f"<div class='card'>{badge} {r.get('description')}<div class='muted'>{r.get('category')}</div></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown("<div class='muted'>No risks</div>", unsafe_allow_html=True)
                    if os_:
                        for o in os_:
                            st.markdown(
                                f"<div class='card'><span class='badge pill'>Obligation</span> {o.get('actor') or 'Actor'} – {o.get('action')}<div class='muted'>Deadline: {o.get('deadline') or 'N/A'}</div></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown("<div class='muted'>No obligations</div>", unsafe_allow_html=True)