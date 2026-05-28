"""
BRAIN — frontend/app.py
═════════════════════════════════════════════════════════════════════════════
Streamlit chat interface + metrics dashboard.

Layout:
    ┌──────────────────────────┬───────────────────────┐
    │                          │  ⚙ Model Selector     │
    │     Chat Interface       │  📊 Betti-1 Holes     │
    │                          │  🧬 Topology Status    │
    │                          │  🔍 Search Triggered   │
    │                          │  ⏱  Latency (ms)      │
    │                          │  📐 Simplices Count    │
    └──────────────────────────┴───────────────────────┘

    streamlit run frontend/app.py --server.port 8501
═════════════════════════════════════════════════════════════════════════════
"""

import time

import httpx
import streamlit as st

# ============================================================================
# 0.  PAGE CONFIG & CONSTANTS
# ============================================================================

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="BRAIN — Medical Reasoning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 1.  CUSTOM CSS — premium dark medical theme
# ============================================================================

st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar styling ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1c 0%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* ── Metric cards ─────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.1);
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    /* ── Chat messages ────────────────────────────────────── */
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── Status badge ─────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-verified {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-void {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-partial {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-unknown {
        background: rgba(107, 114, 128, 0.15);
        color: #9ca3af;
        border: 1px solid rgba(107, 114, 128, 0.3);
    }

    /* ── Header ───────────────────────────────────────────── */
    .brain-header {
        text-align: center;
        padding: 8px 0 16px 0;
        border-bottom: 1px solid rgba(99, 102, 241, 0.15);
        margin-bottom: 20px;
    }
    .brain-header h1 {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .brain-header p {
        color: #64748b;
        font-size: 0.8rem;
        margin: 4px 0 0 0;
    }

    /* ── Divider ──────────────────────────────────────────── */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.12);
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2.  SESSION STATE INITIALISATION
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "betti_1_holes": 0,
        "topology_status": "UNKNOWN",
        "search_attempted": False,
        "execution_time_ms": 0.0,
        "simplices_count": 0,
    }

if "backend_online" not in st.session_state:
    st.session_state.backend_online = False


# ============================================================================
# 3.  BACKEND COMMUNICATION
# ============================================================================

def check_backend_health() -> dict | None:
    """Ping the FastAPI health endpoint."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return None


def send_query(user_query: str, llm_model: str) -> dict | None:
    """POST a query to the FastAPI backend."""
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/query",
            json={"user_query": user_query, "llm_model": llm_model},
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Backend error {resp.status_code}: {resp.text}")
            return None
    except httpx.ConnectError:
        st.error("⚠ Cannot reach backend. Is the FastAPI server running on port 8000?")
        return None
    except httpx.TimeoutException:
        st.error("⚠ Backend request timed out (60s). The LLM may be slow or unavailable.")
        return None


# ============================================================================
# 4.  SIDEBAR — Controls & Dashboard
# ============================================================================

with st.sidebar:
    # Header
    st.markdown("""
    <div class="brain-header">
        <h1>🧠 BRAIN</h1>
        <p>Topological Medical Reasoning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Backend Status ────────────────────────────────────────────
    health = check_backend_health()
    if health:
        st.session_state.backend_online = True
        st.success("● Backend Online", icon="✅")
    else:
        st.session_state.backend_online = False
        st.error("● Backend Offline", icon="🔴")
        st.caption("Start with: `uvicorn backend.api:app --port 8000`")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Model Selector ────────────────────────────────────────────
    st.markdown("### ⚙ Engine Selector")

    MODELS = [
        "groq/llama-3.1-8b-instant",
        "groq/llama-3.3-70b-versatile",
        "groq/gemma2-9b-it",
        "ollama/mistral",
        "ollama/llama3",
        "ollama/gemma2",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.5-flash",
        "gpt-4o-mini",
    ]

    selected_model = st.selectbox(
        "LLM Model",
        options=MODELS,
        index=0,
        help="Switch the reasoning engine per-request via LiteLLM",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Live Metrics Dashboard ────────────────────────────────────
    st.markdown("### 📊 Live Metrics")

    metrics = st.session_state.last_metrics

    # Row 1: Betti-1 & Topology Status
    col1, col2 = st.columns(2)
    with col1:
        betti = metrics["betti_1_holes"]
        st.metric(
            label="Betti-1 Holes",
            value=betti,
            help="Topological loops/voids — higher = more uncertainty",
        )
    with col2:
        status = metrics["topology_status"]
        status_cls = {
            "VERIFIED": "verified",
            "EPISTEMIC_VOID": "void",
            "PARTIAL": "partial",
            "UNKNOWN": "unknown",
        }.get(status, "unknown")

        st.markdown(f"""
        <div style="margin-top: 6px;">
            <span style="color: #94a3b8; font-size: 0.75rem; font-weight: 500;
                         letter-spacing: 0.05em; text-transform: uppercase;">
                Topology Status
            </span><br/>
            <span class="status-badge status-{status_cls}">{status}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Row 2: Search & Latency
    col3, col4 = st.columns(2)
    with col3:
        searched = metrics["search_attempted"]
        st.metric(
            label="Search Triggered",
            value="✅ Yes" if searched else "—  No",
        )
    with col4:
        st.metric(
            label="Latency",
            value=f"{metrics['execution_time_ms']:.0f} ms",
        )

    # Row 3: Simplices
    st.metric(
        label="Simplices (Total)",
        value=f"{metrics['simplices_count']:,}",
        help="Total simplices in the GUDHI SimplexTree",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Brain info from health check ──────────────────────────────
    if health:
        st.markdown("### 🧬 Brain State")
        st.caption(f"Triples ingested: **{health.get('triples_ingested', 0):,}**")
        st.caption(f"Simplices in memory: **{health.get('simplices_count', 0):,}**")
        st.caption(f"Betti-1 (global): **{health.get('betti_1', 0)}**")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Clear chat ────────────────────────────────────────────────
    if st.button("🗑  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================================
# 5.  MAIN COLUMN — Chat Interface
# ============================================================================

# Title
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="font-size: 1.8rem; font-weight: 700; margin: 0;">
        Medical Reasoning Chat
    </h1>
    <p style="color: #64748b; font-size: 0.9rem; margin: 4px 0 0 0;">
        Ask medical questions — BRAIN reasons over a topological knowledge graph
    </p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

        # Show inline metrics for assistant messages
        if msg["role"] == "assistant" and "metrics" in msg:
            m = msg["metrics"]
            with st.expander("📊 Pipeline Metrics", expanded=False):
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Betti-1", m.get("betti_1_holes", 0))
                mc2.metric("Status", m.get("topology_status", "—"))
                mc3.metric("Searched", "Yes" if m.get("search_attempted") else "No")
                mc4.metric("Latency", f"{m.get('execution_time_ms', 0):.0f}ms")

                entities = m.get("entities_found", [])
                if entities:
                    st.caption(f"**Entities detected:** {', '.join(entities)}")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a medical question…", key="chat_input"):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(prompt)

    # Check backend
    if not st.session_state.backend_online:
        with st.chat_message("assistant", avatar="🧠"):
            st.error("Backend is offline. Please start the FastAPI server first.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠ Backend is offline. Please start the FastAPI server.",
        })
    else:
        # Send to backend
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Reasoning through topology…"):
                result = send_query(prompt, selected_model)

            if result:
                # Display answer
                st.markdown(result["final_answer"])

                # Show inline metrics
                with st.expander("📊 Pipeline Metrics", expanded=True):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Betti-1", result.get("betti_1_holes", 0))
                    mc2.metric("Status", result.get("topology_status", "—"))
                    mc3.metric("Searched", "Yes" if result.get("search_attempted") else "No")
                    mc4.metric("Latency", f"{result.get('execution_time_ms', 0):.0f}ms")

                    entities = result.get("entities_found", [])
                    if entities:
                        st.caption(f"**Entities detected:** {', '.join(entities)}")

                # Update sidebar metrics
                st.session_state.last_metrics = {
                    "betti_1_holes": result.get("betti_1_holes", 0),
                    "topology_status": result.get("topology_status", "UNKNOWN"),
                    "search_attempted": result.get("search_attempted", False),
                    "execution_time_ms": result.get("execution_time_ms", 0.0),
                    "simplices_count": result.get("simplices_count", 0),
                }

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["final_answer"],
                    "metrics": result,
                })
            else:
                st.error("Failed to get a response from the backend.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠ Failed to get a response from the backend.",
                })

        st.rerun()
