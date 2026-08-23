#!/bin/bash
# Helper script to run the AI Runtime service

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please create it first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate

# Check if packages are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting AI Runtime service..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
