# Makefile - bootstrap development environment for the Saluki plugin
# Default assumes the framework checkout lives next to this repo at ../biolm_utils
# You can override FRAMEWORK_PATH when calling make, e.g.:
#   make bootstrap FRAMEWORK_PATH=/path/to/biolm_utils

POETRY := poetry
PY := poetry run python

.PHONY: bootstrap install-framework install-plugin deps test example help

help:
	@echo "Make targets:"
	@echo "  bootstrap          -> poetry install; install framework and plugin into poetry venv"
	@echo "  install-framework  -> install framework into the poetry venv (editable)"
	@echo "  install-plugin     -> install this plugin into the poetry venv (editable)"
	@echo "  deps               -> poetry install (create venv and deps)"
	@echo "  test               -> run plugin tests (poetry run pytest)"
	@echo "  example            -> run the quick demo via poetry run"

# Default framework path (can be overridden on the make commandline)
FRAMEWORK_PATH ?= ../biolm_utils

deps:
	# Use --no-root to avoid installing the current project package in this repo
	# (plugin repo may not intend to be installed as a top-level package).
	$(POETRY) install --no-root --no-interaction

install-framework: deps
	@echo "Installing framework (editable) from $(FRAMEWORK_PATH) into Poetry venv..."
	$(PY) -m pip install -e $(FRAMEWORK_PATH)

install-plugin: deps
	@echo "Installing plugin (editable) into Poetry venv..."
	$(PY) -m pip install -e ./saluki_plugin

bootstrap: install-framework install-plugin
	@echo "Bootstrap complete. Framework and plugin are installed into the Poetry venv."

test:
	$(PY) -m pytest -q

example:
	$(PY) examples/quick_train_saluki.py
