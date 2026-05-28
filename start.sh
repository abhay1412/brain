#!/usr/bin/env bash
# ============================================================================
# BRAIN — Start Both Servers  (FastAPI on :8000, Streamlit on :8501)
# ============================================================================
# Usage:  bash start.sh
# Stop:   Ctrl+C  (kills both processes)
# ============================================================================

set -euo pipefail

# Activate venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "brain_env/bin/activate" ]; then
    source brain_env/bin/activate
else
    echo "❌ brain_env not found. Run setup_server.sh first."
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         BRAIN — Starting Server Stack                       ║"
echo "║         FastAPI  →  http://localhost:8000                    ║"
echo "║         Streamlit →  http://localhost:8501                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Ensure brain_data directory exists for checkpoints
mkdir -p brain_data

# ── Trap Ctrl+C to kill both background processes ────────────────────────────
cleanup() {
    echo ""
    echo "🛑 Shutting down …"
    kill $FASTAPI_PID 2>/dev/null || true
    kill $STREAMLIT_PID 2>/dev/null || true
    wait $FASTAPI_PID 2>/dev/null || true
    wait $STREAMLIT_PID 2>/dev/null || true
    echo "✅ All servers stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start FastAPI (1 worker to preserve singletons) ──────────────────────────
echo "▸ Starting FastAPI backend …"
uvicorn backend.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &
FASTAPI_PID=$!
echo "  PID: $FASTAPI_PID"

# Give FastAPI a moment to start loading models before Streamlit boots
sleep 3

# ── Start Streamlit ──────────────────────────────────────────────────────────
echo "▸ Starting Streamlit frontend …"
streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!
echo "  PID: $STREAMLIT_PID"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Both servers running. Press Ctrl+C to stop."
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Wait for either process to exit
wait -n $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true
echo "⚠ A server exited unexpectedly."
cleanup
