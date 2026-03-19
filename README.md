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

## Usage

After installation, Saluki integrates seamlessly with the BioLM framework. Since Saluki is a CNN-based model, it **does not require pre-training** and can be used directly for fine-tuning on labeled data.

### Quick Start Configuration

Create a configuration file (e.g., `config.yaml`) for your experiment:

```yaml
plugin: saluki
outputpath: /tmp/saluki_experiment
task: classification  # or 'regression'
data_source:
  filepath: /path/to/your/data.tsv
  columnsep: "\t"
  idpos: 1
  seqpos: 2
  labelpos: 3
  splitratio: [80, 10, 10]
training:
  nepochs: 10
  batchsize: 8
```

### Training Commands

**Fine-tune on your data:**

```bash
poetry run biolm mode=fine-tune plugin=saluki task=classification data_source.filepath=/path/to/data.tsv outputpath=/tmp/saluki_run
```

**Make predictions:**

```bash
poetry run biolm mode=predict plugin=saluki task=classification data_source.filepath=/path/to/test_data.tsv inference.pretrainedmodel=/path/to/model.safetensors outputpath=/tmp/saluki_run
```

**Interpret features:**

```bash
poetry run biolm mode=interpret plugin=saluki task=classification data_source.filepath=/path/to/data.tsv inference.pretrainedmodel=/path/to/model.safetensors outputpath=/tmp/saluki_run
```

### Data Format

Saluki expects tab-separated data with columns for ID, label, and sequence:

```tsv
ID	Label	Sequence
seq_001	1.5	AUGCUAGCUAGC
seq_002	2.3	AUGGCUAUGGCU
```

- `idpos`: Column index (1-based) for sequence IDs
- `seqpos`: Column index for RNA sequences
- `labelpos`: Column index for labels (numeric for regression, 0/1 for classification)

### Configuration Options

Key configuration parameters for Saluki:

- **Task**: `classification` or `regression`
- **Training**: `nepochs`, `batchsize`, `learning_rate` (default: 0.001)
- **Data**: `splitratio` for train/validation/test splits
- **Interpretation**: `inference.looscores.handletokens` (`mask` or `remove`)

For complete configuration options, see the [BioLM configuration documentation](https://github.com/dieterich-lab/biolm_utils/blob/biolm-2.0/README.md#configuration-management).
