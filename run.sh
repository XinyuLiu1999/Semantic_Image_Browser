#!/bin/bash

# WikiKG Semantic Image Retrieval System
# Startup script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=============================================="
echo "  WikiKG Semantic Image Retrieval"
echo "=============================================="
echo -e "${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# Check if frontend needs to be built
FRONTEND_DIST="$SCRIPT_DIR/frontend/dist"
if [ ! -d "$FRONTEND_DIST" ]; then
    echo -e "${YELLOW}Frontend not built.${NC}"

    if command -v npm &> /dev/null; then
        echo -e "${BLUE}Building frontend...${NC}"
        cd frontend
        npm install
        npm run build
        cd ..
        echo -e "${GREEN}Frontend built successfully.${NC}"
    else
        echo -e "${YELLOW}Node.js not found. Frontend will not be served.${NC}"
        echo -e "${YELLOW}Install Node.js and run 'npm install && npm run build' in the frontend directory.${NC}"
    fi
fi

# Configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo ""
echo -e "${GREEN}=============================================="
echo "  Starting server at http://$HOST:$PORT"
echo "  API docs at http://$HOST:$PORT/docs"
echo "=============================================="
echo -e "${NC}"

# Start the server
cd backend
python main.py "$@"
