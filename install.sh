#!/bin/bash
# install.sh - Saluki Plugin Installation Script
# Installs this plugin into the BioLM framework environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Saluki Plugin Installation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Find the framework
# Check if we are already in a virtual environment that looks like the framework's
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/../pyproject.toml" ] && grep -q "name = \"biolm-utils\"" "$VIRTUAL_ENV/../pyproject.toml"; then
    FRAMEWORK_PATH="$(dirname "$VIRTUAL_ENV")"
    print_info "Detected active framework environment at: $FRAMEWORK_PATH"
else
    FRAMEWORK_PATH="${BIOLM_FRAMEWORK_PATH:-../biolm_utils}"
fi

if [ ! -z "$1" ]; then
    FRAMEWORK_PATH="$1"
fi

print_info "Looking for BioLM framework at: $FRAMEWORK_PATH"

if [ ! -d "$FRAMEWORK_PATH" ]; then
    print_error "Framework not found at: $FRAMEWORK_PATH"
    echo ""
    echo "Options:"
    echo "  1. Specify framework path: $0 /path/to/biolm_utils"
    echo "  2. Set environment variable: export BIOLM_FRAMEWORK_PATH=/path/to/biolm_utils"
    echo "  3. Clone framework to default location: git clone ... ../biolm_utils"
    exit 1
fi

print_success "Framework found"

# Step 2: Check if framework has a virtual environment
if [ ! -d "$FRAMEWORK_PATH/.venv" ]; then
    print_error "Framework not installed (no .venv found)"
    echo ""
    echo "Install the framework first:"
    echo "  cd $FRAMEWORK_PATH"
    echo "  poetry install"
    exit 1
fi

print_success "Framework environment found"

# Step 3: Get absolute paths
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "$FRAMEWORK_PATH" && pwd)"

print_info "Plugin directory: $PLUGIN_DIR"
print_info "Framework directory: $FRAMEWORK_DIR"

# Step 4: Install plugin into framework environment
echo ""
print_info "Installing Saluki plugin into framework environment..."

cd "$FRAMEWORK_DIR"
poetry run pip install -e "$PLUGIN_DIR"

print_success "Plugin installed"

# Step 5: Verify registration
echo ""
print_info "Verifying plugin registration..."

VERIFICATION=$(poetry run python -c "
import importlib.metadata
import sys

try:
    eps = list(importlib.metadata.entry_points(group='biolm.plugins'))
    saluki_found = any(ep.name == 'saluki' for ep in eps)
    
    if saluki_found:
        print('SUCCESS')
        sys.exit(0)
    else:
        print('NOT_FOUND')
        sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(2)
" 2>&1)

if echo "$VERIFICATION" | grep -q "SUCCESS"; then
    print_success "Saluki plugin successfully registered!"
    echo ""
    
    # Show all registered plugins
    cd "$FRAMEWORK_DIR"
    poetry run python -c "
import importlib.metadata
eps = list(importlib.metadata.entry_points(group='biolm.plugins'))
print('Registered plugins:')
for ep in eps:
    print(f'  • {ep.name}')
"
    
    echo ""
    print_success "Installation complete!"
    echo ""
    print_info "You can now use the Saluki plugin:"
    echo "  cd $FRAMEWORK_DIR"
    echo "  poetry run biolm --help"
    
elif echo "$VERIFICATION" | grep -q "NOT_FOUND"; then
    print_error "Plugin installed but not registered as entry point"
    print_info "Check your pyproject.toml for:"
    echo "  [tool.poetry.plugins.\"biolm.plugins\"]"
    echo "  saluki = \"saluki_plugin.config:get_config\""
    exit 1
else
    print_error "Verification failed: $VERIFICATION"
    exit 2
fi
