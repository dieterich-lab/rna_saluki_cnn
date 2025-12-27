# Saluki Plugin for BioLM 2.0

**CNN-based RNA regulatory prediction plugin**

This plugin integrates Saluki, a convolutional neural network, into the BioLM framework for:
- **RNA sequence tokenization**
- **Supervised learning** (classification/regression)
- **Feature interpretation**

## Installation

1. Install the BioLM framework:
   ```bash
   git clone https://github.com/dieterich-lab/biolm_utils.git
   cd biolm_utils
   git checkout biolm-2.0
   ./install.sh
   ```

2. Install the Saluki plugin:
   ```bash
   poetry run biolm install-plugin https://github.com/dieterich-lab/rna_saluki_cnn.git
   ```
