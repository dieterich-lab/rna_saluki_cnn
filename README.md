> **Note:** The `saluki-2.0` branch contains the latest, actively developed version of this plugin. The `main` branch is legacy. For the newest features and code, please [switch to the `saluki-2.0` branch](https://github.com/dieterich-lab/rna_saluki_cnn/tree/saluki-2.0).

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

2. Install the Saluki plugin from the `saluki-2.0` branch:

   ```bash
   poetry run biolm install-plugin "https://github.com/dieterich-lab/rna_saluki_cnn.git?ref=saluki-2.0"
   ```

## Developer install

When making changes to the Saluki plugin, install it directly from your local checkout so Python uses the editable source:

```bash
poetry run biolm develop-plugin /path/to/rna_saluki_cnn
```

## Local commit/push hooks (recommended)

Install local hooks once per clone:

```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```

For hook lifecycle details (`pre-commit` vs `pre-push`) and current stage behavior, see [PLUGIN_CONTRACT.md in biolm_utils](https://github.com/dieterich-lab/biolm_utils/blob/biolm-2.0/docs/PLUGIN_CONTRACT.md#9-local-git-hook-lifecycle-pre-commit-vs-pre-push).
