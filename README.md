# Saluki Plugin for BioLM Utils

This repository provides the **Saluki plugin** for `biolm_utils`, the core framework for tokenizing, pre-training, and fine-tuning language models on biological sequences.

**Note**: This is a plugin, not the main framework. Start with the `biolm_utils` framework repo for installation.

The Saluki plugin includes a CNN-based model (`HFSaluki`) and dataset (`RNACNNDataset`) for RNA sequence analysis.

## Quick Start: Install Framework + Saluki Plugin

The Saluki plugin is built into the `biolm_utils` framework and loads automatically.

**Start here**:

```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
poetry install
```

Then run an experiment:

```bash
poetry run biolm fine-tune --config-path ./biolm/plugins/saluki/exampleconfigs/tokenize_fine-tune.yaml
```

For help: `poetry run biolm --help`

## Manual Plugin Installation (Advanced Users)

If you want to develop the Saluki plugin separately:

```bash
# 1. Clone and install the framework first
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
poetry install

# 2. Clone the plugin repository
git clone https://github.com/dieterich-lab/rna_saluki_cnn.git
cd rna_saluki_cnn

# 3. Install the plugin in editable/development mode
poetry install

# The plugin registers itself via entry points and is automatically discovered
```

## Plugin Configuration

The Saluki plugin is configured via the `saluki_plugin/config.py` module, which defines:

- **Model**: `HFSaluki` - CNN-based model for RNA sequence classification/regression
- **Dataset**: `RNACNNDataset` - Loads RNA sequences and labels
- **Pretraining**: Not required (model trains from scratch on labeled data)
- **Fine-tuning**: Supported for regression and classification tasks

## Running Experiments

Use Hydra-based config files to run experiments:

```bash
poetry run biolm fine-tune \
  --config-path ./biolm/plugins/saluki/exampleconfigs \
  data_source.filepath=/path/to/your/data.txt \
  task=regression
```

See `./biolm/plugins/saluki/exampleconfigs/` for example configurations.

## File Structure

```
biolm/plugins/saluki/
├── rna_cnn_dataset.py      # RNACNNDataset implementation
├── rna_cnn_models.py       # HFSaluki model implementation
├── exampleconfigs/         # Example Hydra configurations
│   ├── tokenize_fine-tune.yaml
│   └── predict_interpret.yaml
└── tests/                  # Plugin tests
```
