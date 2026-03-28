"""
AI-Based Fake Financial News Detection - Streamlit Web Application
Premium dark-themed UI with glassmorphism, animations, and confidence gauges
"""

import os
import sys
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.predictor import NewsClassifier

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake Financial News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global Theme ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 25%, #11192e 50%, #0d1321 75%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide Streamlit Defaults (keep sidebar toggle visible) ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important; visibility: visible !important;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}

    /* ── Force sidebar always visible ── */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        width: 300px !important;
        min-width: 300px !important;
        transform: none !important;
        position: relative !important;
    }
    [data-testid="stSidebar"] > div {
        width: 300px !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
        border: 1px solid rgba(99,102,241,0.3);
        color: #a78bfa;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0e7ff 0%, #a78bfa 30%, #818cf8 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 0.75rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    }

    /* ── Result Cards ── */
    .result-real {
        background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(52,211,153,0.05));
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: scaleIn 0.5s ease-out;
    }
    .result-fake {
        background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(248,113,113,0.05));
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: scaleIn 0.5s ease-out;
    }
    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-label-real { color: #34d399; }
    .result-label-fake { color: #f87171; }
    .result-confidence {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .result-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }

    /* ── Confidence Bar ── */
    .confidence-bar-container {
        width: 100%;
        height: 10px;
        background: rgba(30,41,59,0.8);
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .confidence-bar-fill-real {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
        transition: width 1s ease-out;
        animation: barGrow 1s ease-out;
    }
    .confidence-bar-fill-fake {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ef4444, #f87171, #fca5a5);
        transition: width 1s ease-out;
        animation: barGrow 1s ease-out;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99,102,241,0.3);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    /* ── Pipeline Steps ── */
    .step-card {
        background: rgba(15,23,42,0.4);
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .step-card:hover {
        border-color: rgba(99,102,241,0.3);
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(99,102,241,0.08);
    }
    .step-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .step-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.25rem;
    }
    .step-desc {
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.4;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
        border-right: 1px solid rgba(99,102,241,0.1);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8;
    }

    /* ── Sample Button ── */
    .sample-btn {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 8px;
        padding: 0.75rem;
        color: #c7d2fe;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.82rem;
        line-height: 1.4;
        margin-bottom: 0.5rem;
    }
    .sample-btn:hover {
        background: rgba(99,102,241,0.15);
        border-color: rgba(99,102,241,0.4);
    }

    /* ── Text Area ── */
    .stTextArea textarea {
        background: rgba(15,23,42,0.6) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    }

    /* ── Button Styling ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.35) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #475569;
        font-size: 0.8rem;
        border-top: 1px solid rgba(99,102,241,0.08);
        margin-top: 3rem;
    }
    .footer a { color: #818cf8; text-decoration: none; }

    /* ── Animations ── */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes barGrow {
        from { width: 0%; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* ── Divider ── */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
        margin: 2rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    """Load the trained model (cached for performance)"""
    classifier = NewsClassifier()
    classifier.load_model()
    return classifier


# ─────────────────────────────────────────────
# SAMPLE NEWS FOR SIDEBAR
# ─────────────────────────────────────────────
SAMPLE_REAL_NEWS = [
    "RBI keeps repo rate unchanged at 6.5% in its latest monetary policy committee meeting. The decision was taken considering the current inflation trajectory.",
    "BSE Sensex closes at record high of 75,000 points, driven by strong buying in banking and IT stocks. FIIs were net buyers worth Rs 3,500 crore.",
    "SEBI introduces new regulations for mutual fund investments requiring fund houses to disclose portfolio holdings on a fortnightly basis.",
    "India's GDP growth rate reaches 7.6% in Q2, driven by strong performance in manufacturing and services sectors.",
    "SBI reports record net profit of Rs 18,500 crore in Q3, representing a 35% year-on-year growth with improving asset quality.",
]

SAMPLE_FAKE_NEWS = [
    "BREAKING: RBI to ban all bank transactions for 30 days! All ATM withdrawals, UPI payments will be suspended. Withdraw all money immediately!",
    "Government to seize all fixed deposits above Rs 5 lakh from banks secretly. This law takes effect from midnight tonight.",
    "Stock market will crash 90% tomorrow, says top analyst. Investors advised to sell all shares immediately. BSE and NSE will shut down permanently.",
    "SBI offering 25% interest rate on savings accounts! Five times higher than any other bank. Limited time offer only!",
    "All bank account holders will receive Rs 1 crore free from the government. Forward this to 10 friends and share bank details to claim.",
]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
# Initialize session state for sample selection
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = ""

with st.sidebar:
    st.markdown("## 🔬 Quick Test Samples")
    st.markdown("---")
    
    st.markdown("#### ✅ Real News Examples")
    for i, news in enumerate(SAMPLE_REAL_NEWS):
        if st.button(f"📰 Sample {i+1}", key=f"real_{i}", use_container_width=True):
            st.session_state.selected_sample = news
        st.caption(news[:80] + "...")
        st.markdown("")
    
    st.markdown("---")
    st.markdown("#### ❌ Fake News Examples")
    for i, news in enumerate(SAMPLE_FAKE_NEWS):
        if st.button(f"⚠️ Sample {i+1}", key=f"fake_{i}", use_container_width=True):
            st.session_state.selected_sample = news
        st.caption(news[:80] + "...")
        st.markdown("")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **AI Fake Financial News Detector**
    
    BCA Final Year Project using 
    **BERT Sentence Embeddings** 
    and Machine Learning to classify 
    financial news as Real or Fake.
    
    **Tech Stack:**
    - 🐍 Python 3.9+
    - 🧠 BERT (all-MiniLM-L6-v2)
    - 🤖 scikit-learn
    - 📝 NLTK
    - 🌐 Streamlit
    """)


# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🤖 AI-Powered Detection Engine</div>
    <div class="hero-title">Fake Financial News<br>Detector</div>
    <div class="hero-subtitle">
        Leverage advanced NLP and Machine Learning to instantly verify 
        financial news articles. Get real-time classification with 
        confidence scores — designed for the Indian financial ecosystem.
    </div>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STATS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">NLP</div>
        <div class="stat-label">Text Analysis</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">ML</div>
        <div class="stat-label">Classification</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">&lt;3s</div>
        <div class="stat-label">Response Time</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">90%+</div>
        <div class="stat-label">Accuracy</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN INPUT SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📰 Enter Financial News Article</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="Paste your financial news article or headline below:",
    value=st.session_state.selected_sample,
    height=180,
    placeholder="Example: RBI keeps repo rate unchanged at 6.5% in its latest monetary policy committee meeting...",
    label_visibility="collapsed"
)

# Analyze button
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    analyze_clicked = st.button("🔍  Analyze News", use_container_width=True)


# ─────────────────────────────────────────────
# PREDICTION & RESULTS
# ─────────────────────────────────────────────
if analyze_clicked:
    if not user_input or user_input.strip() == "":
        st.markdown("""
        <div class="glass-card" style="text-align: center; border-color: rgba(251,191,36,0.3);">
            <div style="font-size: 2.5rem;">⚠️</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #fbbf24; margin: 0.5rem 0;">
                No Input Provided
            </div>
            <div style="color: #94a3b8;">
                Please paste a financial news article or headline in the text area above.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            # Load classifier and predict
            with st.spinner("🔍 Analyzing article..."):
                classifier = load_classifier()
                result = classifier.predict(user_input)
            
            confidence_pct = result.confidence * 100
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if result.label == "REAL":
                st.markdown(f"""
                <div class="result-real">
                    <div class="result-emoji">✅</div>
                    <div class="result-label result-label-real">REAL NEWS</div>
                    <div class="result-confidence">
                        Confidence Score: <strong>{confidence_pct:.1f}%</strong>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-fill-real" style="width: {confidence_pct}%;"></div>
                    </div>
                    <div style="color: #6ee7b7; font-size: 0.85rem; margin-top: 0.5rem;">
                        ✓ This article appears to be legitimate financial news
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fake">
                    <div class="result-emoji">🚨</div>
                    <div class="result-label result-label-fake">FAKE NEWS</div>
                    <div class="result-confidence">
                        Confidence Score: <strong>{confidence_pct:.1f}%</strong>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-fill-fake" style="width: {confidence_pct}%;"></div>
                    </div>
                    <div style="color: #fca5a5; font-size: 0.85rem; margin-top: 0.5rem;">
                        ⚠ This article appears to contain misleading or fake financial information
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Prediction details
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 Prediction Details", expanded=False):
                det_col1, det_col2, det_col3 = st.columns(3)
                with det_col1:
                    st.metric("Classification", result.label)
                with det_col2:
                    st.metric("Confidence", f"{confidence_pct:.1f}%")
                with det_col3:
                    st.metric("Timestamp", result.timestamp)
                
                st.markdown("**Analyzed Text Preview:**")
                st.code(user_input[:500] + ("..." if len(user_input) > 500 else ""), language=None)
                
        except FileNotFoundError as e:
            st.error(f"⚠️ Model not found! Please train the model first by running:\n\n`python src/models/train_model.py`\n\nError: {e}")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


# ─────────────────────────────────────────────
# HOW IT WORKS SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">⚙️ How It Works</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

step_cols = st.columns(5)
steps = [
    ("📝", "Input", "Paste financial news article or headline"),
    ("🔧", "Preprocess", "Tokenization, stopword removal, lemmatization"),
    ("🧠", "BERT Embeddings", "Sentence-level encoding via all-MiniLM-L6-v2"),
    ("🤖", "ML Classification", "Random Forest classifier on BERT features"),
    ("✅", "Result", "REAL/FAKE label with confidence score"),
]

for col, (icon, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-icon">{icon}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TECH STACK SECTION
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tech_cols = st.columns(5)
techs = [
    ("🧠", "BERT", "Sentence Embeddings"),
    ("🐍", "Python", "Core Language"),
    ("🤖", "scikit-learn", "ML Engine"),
    ("📝", "NLTK", "NLP Preprocessing"),
    ("🌐", "Streamlit", "Web Framework"),
]

for col, (icon, name, role) in zip(tech_cols, techs):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 1.8rem;">{icon}</div>
            <div style="font-size: 1rem; font-weight: 700; color: #e2e8f0; margin: 0.25rem 0;">{name}</div>
            <div class="stat-label">{role}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>
        <strong>AI-Based Fake Financial News Detection System</strong><br>
        BCA Final Year Project 2025-2026<br>
        Built with ❤️ using BERT, Python, scikit-learn, NLTK & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
