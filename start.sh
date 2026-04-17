#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  DocumentLens NLP Summarizer – Setup & Run Script
#  Works on macOS, Linux, Windows (Git Bash / WSL)
# ──────────────────────────────────────────────────────────────────────────────

set -e

PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    PYTHON="$cmd"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "❌  Python not found. Please install Python 3.9+ from https://python.org"
  exit 1
fi

VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅  Found Python $VERSION"

echo ""
echo "📦  Creating virtual environment..."
$PYTHON -m venv venv

echo "📦  Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  source venv/Scripts/activate
else
  source venv/bin/activate
fi

echo "📦  Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✅  Setup complete! Starting server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   🌐  Open your browser at: http://127.0.0.1:5000"
echo "   ⌨   Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python app.py
