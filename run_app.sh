#!/usr/bin/env bash
# Start the investor chat (macOS / Linux):   ./run_app.sh
# First run: finds Python 3.12+, creates .venv and installs the requirements (a few minutes).
# If an earlier install failed, it is repeated: the marker .venv/installed.ok is written only on success.
set -e
cd "$(dirname "$0")"

ok_version() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; }

if [ -x .venv/bin/python ] && ! ok_version .venv/bin/python; then
  echo "The existing .venv uses Python older than 3.12. Rebuilding it..."
  rm -rf .venv
fi

if [ ! -f .venv/installed.ok ]; then
  PY=""
  for c in python3.14 python3.13 python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && ok_version "$c"; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.12 or newer was not found. The pinned packages (numpy 2.5, pandas 3,"
    echo "scikit-learn 1.9) need it. Install Python 3.13 (https://www.python.org/downloads/) and run again."
    exit 1
  fi
  [ -x .venv/bin/python ] || { echo "Creating .venv with $PY"; "$PY" -m venv .venv; }
  echo "Installing the requirements (first run only, a few minutes)..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-app.txt
  touch .venv/installed.ok
fi
exec .venv/bin/python -m streamlit run src/app/streamlit_app.py
