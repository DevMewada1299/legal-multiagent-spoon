import streamlit as st
import asyncio
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from app.ingestion import PDFIngestor
from app.classification import LegalClassifier
from app.extraction import ObligationExtractor
from app.retrieval import FaissRetriever
from app.agents import LegalAgentOrchestrator
import importlib

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* Clause cards with dark theme */
    .clause-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .clause-card h4 {
        color: #e2e8f0;
        margin-bottom: 1rem;
    }

    .clause-info {
        background: #0f172a;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .info-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .info-item {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .info-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .info-value {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .clause-text {
        background: #0f172a;
        color: #cbd5e1;
        padding: 1.25rem;
        border-radius: 8px;
        line-height: 1.7;
        border: 1px solid #334155;
    }

    .obligation-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
    }

    .no-obligation-badge {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.3);
    }

    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #a78bfa;
        color: #e2e8f0;
        line-height: 1.8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        height: 200px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }

    .feature-card h4 {
        margin-top: 0;
        margin-bottom: 1rem;
    }

    .feature-card p {
        color: #94a3b8;
    }

    /* Getting started box */
    .getting-started {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-top: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }

    .getting-started h3 {
        margin-top: 0;
        color: white;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: #1e293b;
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        color: #94a3b8;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa;
    }

    /* Text area styling */
    .stTextArea textarea {
        background: #0f172a !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ AI Legal Document Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered analysis of legal documents with clause classification, obligation extraction, and intelligent summarization</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key to enable AI agents for classification, extraction, and summarization")

    st.divider()

    st.header("📊 About")
    st.markdown("""
    This tool uses **Spoon OS Multi-Agent Framework** with Google Gemini for comprehensive legal document analysis:

    - **📖 Document Ingestion**: Extracts text and clauses from PDFs
    - **🤖 Classification Agent**: Spoon OS agent categorizes legal clauses by type
    - **🤖 Extraction Agent**: Spoon OS agent identifies binding obligations
    - **🔎 Smart Retrieval**: Semantic search using vector embeddings
    - **🤖 Summarization Agent**: Spoon OS agent generates structured summaries

    **Powered by Spoon OS** - An agentic operating system for building sentient, composable, and interoperable AI agents.

    *Requires Gemini API key for AI agents. Falls back to rule-based analysis if not provided.*
    """)

    st.divider()

    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
        <p>Built with Streamlit & AI</p>
        <p>© 2024 AI Legal Document Analyzer</p>
    </div>
    """, unsafe_allow_html=True)

async def process_document(file_path: str, api_key: str):
    """Process the uploaded PDF document through the multi-agent analysis pipeline."""

    # 1. Ingestion
    with st.spinner("📖 Reading PDF document..."):
        ingestor = PDFIngestor()
        clauses = ingestor.ingest(file_path)
        st.success(f"✓ Extracted {len(clauses)} clauses from the document")

    # Check if API key is provided for AI agents
    if not api_key:
        st.warning("⚠️ No Gemini API key provided. Using basic rule-based analysis.")
        use_ai_agents = False
    else:
        use_ai_agents = True

    classified_clauses = clauses
    extracted_clauses = clauses
    summary = None

    try:
        if use_ai_agents:
            # Initialize Spoon OS Agent Orchestrator
            with st.spinner("🚀 Initializing Spoon OS Multi-Agent System..."):
                orchestrator = LegalAgentOrchestrator()
                await orchestrator.initialize(api_key)
                st.success("✓ Spoon OS Agents initialized")

            # 2. Classification Agent (Spoon OS)
            with st.spinner("🤖 Spoon OS Classification Agent analyzing clauses..."):
                classified_clauses = await orchestrator.classify_clauses(clauses)
                st.success("✓ Classification Agent completed")

            # 3. Extraction Agent (Spoon OS)
            with st.spinner("🤖 Spoon OS Extraction Agent identifying obligations..."):
                extracted_clauses = await orchestrator.extract_obligations(classified_clauses)
                st.success("✓ Extraction Agent completed")

            # 4. Summarization Agent (Spoon OS)
            with st.spinner("🤖 Spoon OS Summarization Agent creating document summary..."):
                summary = await orchestrator.generate_summary(extracted_clauses)
                st.success("✓ Summarization Agent completed")

        else:
            # Fallback to rule-based analysis
            with st.spinner("🏷️ Classifying legal clauses (rule-based)..."):
                classifier = LegalClassifier()
                classified_clauses = classifier.classify(clauses)
                st.success("✓ Clauses classified")

            with st.spinner("🔍 Extracting obligations (rule-based)..."):
                extractor = ObligationExtractor()
                extracted_clauses = extractor.extract(classified_clauses)
                st.success("✓ Obligations extracted")

    except Exception as e:
        st.error(f"❌ Agent processing failed: {str(e)}")
        st.info("💡 Falling back to basic rule-based analysis...")

        # Fallback to rule-based
        classifier = LegalClassifier()
        classified_clauses = classifier.classify(clauses)

        extractor = ObligationExtractor()
        extracted_clauses = extractor.extract(classified_clauses)

    # 5. Retrieval (always runs, doesn't require API key)
    with st.spinner("🔎 Indexing clauses for retrieval..."):
        retriever = FaissRetriever()
        retriever.index(extracted_clauses)
        retrieved_clauses = retriever.retrieve("What are the termination conditions?")
        st.success("✓ Clauses indexed and retrieved")

    return extracted_clauses, retrieved_clauses, summary

def main():
    # File uploader with enhanced UI
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a legal document in PDF format (Max 200MB)",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Display file details in a nice card
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**📎 File:** {uploaded_file.name}")
        with col2:
            st.markdown(f"**📏 Size:** {uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.markdown(f"**📋 Type:** PDF")

        st.divider()

        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Process button - centered and prominent
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🚀 Analyze Document", type="primary", use_container_width=True)

        if analyze_button:
            try:
                # Run the async processing
                extracted_clauses, retrieved_clauses, summary = asyncio.run(
                    process_document(tmp_file_path, api_key_input)
                )

                # Display metrics
                st.markdown("---")
                st.markdown("### 📊 Analysis Overview")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Clauses", len(extracted_clauses))
                with col2:
                    obligation_count = sum(1 for c in extracted_clauses if c.get('has_obligation'))
                    st.metric("With Obligations", obligation_count)
                with col3:
                    st.metric("Retrieved Matches", len(retrieved_clauses))
                with col4:
                    st.metric("Summary", "✓ Generated" if summary else "✗ N/A")

                st.markdown("---")

                # Use tabs for better organization
                tab1, tab2, tab3 = st.tabs(["📝 Summary", "🔍 Retrieved Clauses", "📋 All Clauses"])

                with tab1:
                    if summary:
                        st.markdown("#### 📊 AI-Generated Document Summary")
                        st.markdown(f"""
                        <div class="summary-box">
                            {summary}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("💡 Summary unavailable. Enter a valid Gemini API key in the sidebar to enable all AI agents (Classification, Extraction, and Summarization).")

                with tab2:
                    st.markdown("#### 🔎 Clauses Relevant to: *What are the termination conditions?*")
                    if retrieved_clauses:
                        for i, clause in enumerate(retrieved_clauses, 1):
                            obligation_status = clause.get('has_obligation', False)
                            badge_class = "obligation-badge" if obligation_status else "no-obligation-badge"
                            badge_text = "✓ Has Obligation" if obligation_status else "No Obligation"

                            with st.container():
                                st.markdown(f"""
                                <div class="clause-card">
                                    <h4>🔖 Clause {i}</h4>
                                    <div class="clause-info">
                                        <div class="info-row">
                                            <div class="info-item">
                                                <span class="info-label">Clause ID</span>
                                                <span class="info-value">{clause.get('id', 'N/A')}</span>
                                            </div>
                                            <div class="info-item">
                                                <span class="info-label">Category</span>
                                                <span class="info-value">{clause.get('label', 'N/A')}</span>
                                            </div>
                                            <div class="info-item">
                                                <span class="info-label">Page</span>
                                                <span class="info-value">{clause.get('page', 'N/A')}</span>
                                            </div>
                                            <div class="info-item">
                                                <span class="info-label">Relevance Score</span>
                                                <span class="info-value">{clause.get('score', 'N/A')}</span>
                                            </div>
                                            <div class="info-item">
                                                <span class="info-label">Status</span>
                                                <span class="{badge_class}">{badge_text}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="clause-text">
                                        {clause.get('text', 'No text available')}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No relevant clauses retrieved for this query.")

                with tab3:
                    st.markdown("#### 📋 All Classified Clauses")
                    if extracted_clauses:
                        st.markdown(f"*Showing all {len(extracted_clauses)} clauses*")
                        st.markdown("<br>", unsafe_allow_html=True)

                        for i, clause in enumerate(extracted_clauses, 1):
                            obligation_status = clause.get('has_obligation', False)
                            badge_class = "obligation-badge" if obligation_status else "no-obligation-badge"
                            badge_text = "✓ Has Obligation" if obligation_status else "No Obligation"

                            with st.container():
                                st.markdown(f"""
                                <div class="clause-card">
                                    <h4>🔖 Clause {i}</h4>
                                    <div class="clause-info">
                                        <div class="info-row">
                                            <div class="info-item">
                                                <span class="info-label">Clause ID</span>
                                                <span class="info-value">{clause.get('id', 'N/A')}</span>
                                            </div>
                                            <div class="info-item">
                                                <span class="info-label">Category</span>
                                                <span class="info-value">{clause.get('label', 'N/A')}</span>
                                            </div>
                                            <div class="info-item">
                                                <span class="info-label">Page</span>
                                                <span class="info-value">{clause.get('page', 'N/A')}</span>
                                            </div>
                                            <div class="info-item">
                                                <span class="info-label">Status</span>
                                                <span class="{badge_class}">{badge_text}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="clause-text">
                                        {clause.get('text', 'No text available')}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No clauses found in the document.")

            except Exception as e:
                st.error(f"❌ An error occurred while processing the document")
                with st.expander("View Error Details"):
                    st.exception(e)

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

    else:
        # Enhanced instructions when no file is uploaded
        st.markdown("""
        <div class="getting-started">
            <h3>🚀 Getting Started</h3>
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">Analyze your legal documents in three simple steps:</p>
            <ol style="font-size: 1rem; line-height: 2;">
                <li><strong>Upload</strong> your PDF legal document using the file uploader above</li>
                <li><strong>Configure</strong> your Gemini API key in the sidebar to enable AI agents</li>
                <li><strong>Analyze</strong> and view comprehensive results including classifications, obligations, and summaries</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #60a5fa;">📖 Smart Extraction</h4>
                <p>Automatically extracts and parses legal clauses from PDF documents with high accuracy.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #34d399;">🤖 Spoon OS Agents</h4>
                <p>Three specialized Spoon OS agents orchestrated for Classification, Extraction, and Summarization.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4 style="color: #a78bfa;">🔍 Intelligent Search</h4>
                <p>Semantic search to retrieve relevant clauses based on natural language queries.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
