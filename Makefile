# Makefile - development environment for the Saluki plugin
# This Makefile supports the new BioLM 2.0 plugin architecture
# where plugins are separate packages with entry point registration.

POETRY := poetry
PY := poetry run python
FRAMEWORK_PATH ?= ../biolm_utils

.PHONY: help install install-dev install-framework test verify-plugin clean

help:
	@echo "Saluki Plugin - Development Makefile"
	@echo "======================================"
	@echo ""
	@echo "Installation targets:"
	@echo "  install            -> Install plugin in current environment"
	@echo "  install-dev        -> Install plugin + development dependencies"
	@echo "  install-framework  -> Install framework with plugin support"
	@echo "  bootstrap          -> Complete setup: framework + plugins"
	@echo ""
	@echo "Development targets:"
	@echo "  test               -> Run plugin tests"
	@echo "  verify-plugin      -> Verify entry point registration"
	@echo "  clean              -> Remove build artifacts and cache"
	@echo ""
	@echo "Configuration:"
	@echo "  FRAMEWORK_PATH=$(FRAMEWORK_PATH)"

install:
	@echo "Installing Saluki plugin..."
	$(POETRY) install --no-interaction

install-dev: install
	@echo "Installing development dependencies..."
	$(POETRY) install --no-interaction --with dev

install-framework:
	@echo "Installing framework with plugin support from $(FRAMEWORK_PATH)..."
	@test -d "$(FRAMEWORK_PATH)" || (echo "Error: Framework not found at $(FRAMEWORK_PATH)" && exit 1)
	cd $(FRAMEWORK_PATH) && $(POETRY) install --no-interaction --with plugins

bootstrap: install-dev install-framework
	@echo ""
	@echo "✅ Bootstrap complete!"
	@echo "   - Saluki plugin installed"
	@echo "   - Framework installed with plugin support"
	@echo ""
	@echo "Verifying plugin registration..."
	@$(MAKE) verify-plugin

test:
	@echo "Running Saluki plugin tests..."
	$(POETRY) run pytest -xvs tests/

verify-plugin:
	@echo "Verifying Saluki plugin entry point..."
	@$(PY) -c "\
	import importlib.metadata; \
	eps = {ep.name: ep.value for ep in importlib.metadata.entry_points(group='biolm.plugins')}; \
	assert 'saluki' in eps, 'Saluki plugin not registered'; \
	assert 'saluki_plugin' in eps['saluki'], f'Wrong module: {eps[\"saluki\"]}'; \
	print('✅ Saluki plugin correctly registered as:', eps['saluki'])"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete"
