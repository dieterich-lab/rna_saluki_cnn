# Saluki Plugin for BioLM 2.0

CNN-based RNA regulatory prediction plugin for the BioLM framework. Part of the BioLM 2.0 plugin architecture.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

**Saluki** is a convolutional neural network for RNA sequence analysis, optimized for regulatory element prediction. This plugin integrates with BioLM 2.0 for:

- **RNA sequence tokenization** (atomic nucleotide encoding)
- **Supervised learning** (classification/regression tasks)
- **Feature interpretation** (leave-one-out analysis)

**Key features:**
- ✅ Long sequences (12K nucleotides)
- ✅ No pre-training required (trains from scratch)
- ✅ Atomic tokenization (a, t, g, c encoding)
- ✅ Interpretable predictions

---

## 📋 Prerequisites

- **BioLM Framework** (required) - [Install guide](https://github.com/dieterich-lab/biolm_utils)
- Python 3.10+
- Poetry

---

## 🚀 Installation

### 1. Install BioLM Framework First

```bash
# Clone and install framework
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
git checkout biolm-2.0
./install.sh
```

### 2. Install Saluki Plugin

```bash
biolm install-plugin https://github.com/dieterich-lab/rna_saluki_cnn.git
```

The plugin automatically:
- Finds the framework
- Installs itself into the framework's environment
- Registers via entry points
- Verifies successful registration

### 3. Verify Installation

```bash
biolm list-plugins
```

---

## 📊 Data Format

Saluki expects **tab-separated** input with comma-separated nucleotides:

```
seq_id       label    sequence
seq_001      1.5      a,t,g,c,a,g,t,c,...
seq_002      2.3      a,t,g,c,a,g,t,c,...
```

**Column specifications:**
- Column 1: Sequence ID
- Column 2: Label (regression value or class)
- Column 3: Comma-separated nucleotides (lowercase: a, t, g, c)

**Important:**
- Columns are **1-indexed** (1, 2, 3...)
- Use `data_source.column_ids=[1,2,3]` in config
- Max sequence length: 12,288 nucleotides

---

## ⚙️ Configuration

Create a config file (e.g., `my_experiment/config.yaml`):

```yaml
# Essential parameters
plugin: saluki                           # Use Saluki plugin
task: regression                         # regression or classification
outputpath: /path/to/results            # Output directory

# Data source
data_source:
  filepath: /path/to/data.txt           # Your RNA data
  column_ids: [1, 2, 3]                 # ID, label, sequence columns
  splitratio: [70, 15, 15]              # Train/val/test split

# Saluki-specific
training:
  blocksize: 12288                       # Required for Saluki
  batchsize: 8                           # Adjust for GPU memory
  num_epochs: 10
  learning_rate: 0.001

# Model architecture
model:
  num_layers: 5                          # CNN layers
  kernel_size: 9                         # Convolution kernel
  num_filters: 256                       # Filters per layer

# Hardware
debugging:
  accelerator: gpu                       # or cpu
  devices: 1
```

**📖 Full config reference:** [BioLM Configuration Guide](https://github.com/dieterich-lab/biolm_utils/blob/biolm-2.0/docs/CONFIGURATION.md)

---

## 🏃 Quick Start

### 1. Tokenization

```bash
cd biolm_utils
poetry run biolm tokenize \
  --config-path ../rna_saluki_cnn/my_experiment \
  plugin=saluki \
  data_source.filepath=/path/to/data.txt
```

### 2. Fine-tuning

```bash
poetry run biolm fine-tune \
  --config-path ../rna_saluki_cnn/my_experiment \
  plugin=saluki \
  task=regression
```

### 3. Prediction

```bash
poetry run biolm predict \
  --config-path ../rna_saluki_cnn/my_experiment \
  plugin=saluki
```

### 4. Interpretation

```bash
poetry run biolm interpret \
  --config-path ../rna_saluki_cnn/my_experiment \
  plugin=saluki
```

---

## 🧬 Model Architecture

**Saluki CNN:**
- **Input:** Atomic nucleotide encoding (a=1, t=2, g=3, c=4)
- **Layers:** 5 convolutional layers (configurable)
- **Kernel size:** 9 (default, configurable)
- **Filters:** 256 per layer (configurable)
- **Output:** Single value (regression) or class probabilities

**Why CNN for RNA:**
- Captures local sequence motifs
- Position-invariant feature detection
- Efficient for long sequences (12K+)
- No attention overhead

---

## 🧪 Testing

```bash
cd rna_saluki_cnn

# Run all plugin tests
poetry run pytest tests/

# Run specific tests
poetry run pytest tests/test_saluki_full_pipeline.py      # End-to-end
poetry run pytest tests/test_saluki_plugin_config.py      # Configuration
```

**Test coverage:**
- ✅ Full pipeline (tokenize → train → test)
- ✅ Plugin discovery and loading
- ✅ Configuration validation
- ✅ Model instantiation

---

## 📂 Project Structure

```
rna_saluki_cnn/
├── saluki_plugin/           # Plugin implementation
│   ├── __init__.py
│   ├── config.py           # Plugin configuration
│   ├── rna_cnn_dataset.py  # RNACNNDataset class
│   └── rna_cnn_models.py   # HFSaluki model
├── tests/                   # Plugin tests
│   ├── test_saluki_full_pipeline.py
│   └── test_saluki_plugin_config.py
├── docs/                    # Documentation (future)
├── pyproject.toml          # Plugin metadata & dependencies
└── README.md               # This file
```

---

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone plugin repo
git clone https://github.com/dieterich-lab/rna_saluki_cnn.git
cd rna_saluki_cnn

# Install in development mode
poetry install --with dev

# Run tests
poetry run pytest tests/ -v

# Check code style
ruff check saluki_plugin/
```

### Plugin Entry Point

The plugin registers itself via `pyproject.toml`:

```toml
[tool.poetry.plugins."biolm.plugins"]
saluki = "saluki_plugin.config:get_config"
```

This allows BioLM to discover Saluki automatically.

**📖 Plugin development guide:** [BioLM Plugin Development](https://github.com/dieterich-lab/biolm_utils/blob/biolm-2.0/docs/PLUGIN_DEVELOPMENT.md)

---

## 🆚 Saluki vs XLNet

| Feature | Saluki | XLNet |
|---------|--------|-------|
| **Architecture** | CNN | Transformer |
| **Sequences** | RNA | Protein |
| **Max Length** | 12,288 tokens | 512 tokens |
| **Pre-training** | ❌ Not required | ✅ Required |
| **Speed** | Fast (CNN) | Slower (attention) |
| **Use Case** | RNA regulatory | Protein function |
| **Interpretability** | Direct (conv filters) | Attention weights |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `poetry run pytest tests/`
5. Check code style: `ruff check .`
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open Pull Request

---

## 📝 Citation

```bibtex
@software{saluki2024,
  title = {Saluki: CNN-based RNA Regulatory Prediction Plugin for BioLM},
  author = {Dieterich Lab},
  year = {2024},
  url = {https://github.com/dieterich-lab/rna_saluki_cnn}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🔗 Related Projects

- **[BioLM Framework](https://github.com/dieterich-lab/biolm_utils)** - Core framework
- **[XLNet Plugin](https://github.com/dieterich-lab/rna_protein_xlnet)** - Protein transformer plugin

---

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/dieterich-lab/rna_saluki_cnn/issues)
- **Framework Docs:** [BioLM Documentation](https://github.com/dieterich-lab/biolm_utils/tree/biolm-2.0/docs)

---

**Built with ❤️ by the Dieterich Lab**
