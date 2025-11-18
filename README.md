# RNA Saluki CNN: Deep Learning for RNA Sequence Analysis

A high-performance deep learning framework for RNA sequence analysis using convolutional neural networks (CNNs). Built on the [Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) architecture, this tool enables:

- **Tokenization**: Convert RNA sequences to one-hot encoded representations
- **Fine-tuning**: Train regression and classification models on RNA data
- **Prediction**: Apply trained models to new sequences
- **Interpretation**: Generate leave-one-out (LOO) scores for model interpretability

## Quick Start

### Installation

```bash
# Clone the repository (biolm_utils is now included as a subtree)
git clone https://github.com/dieterich-lab/rna_saluki_cnn.git
cd rna_saluki_cnn
python3 -m venv ~/.venvs/rna_saluki
source ~/.venvs/rna_saluki/bin/activate
pip install pipenv
pipenv install
```

**Requirements:** Python 3.8+, PyTorch, CUDA (optional for GPU acceleration)

Optional: keep the included `biolm_utils` subtree up-to-date with upstream

 
```bash
# add upstream (only once, if missing)
git remote add biolm_upstream https://github.com/dieterich-lab/biolm_utils.git || true
# pull latest from upstream into the subtree
git fetch biolm_upstream
git subtree pull --prefix=biolm_utils biolm_upstream main --squash
```


If you make changes in `biolm_utils` locally and want to push them upstream (you need write access to upstream), use:

 
```bash
git subtree push --prefix=biolm_utils biolm_upstream main
```


**GPU Configuration:**

```bash
# Force CPU usage
python saluki.py debugging.accelerator=cpu ...

# Use GPU (default)
python saluki.py debugging.accelerator=gpu ...
```

### Basic Usage

```bash
# Tokenize your RNA sequences
python saluki.py mode=tokenize data_source.filepath=my_sequences.txt

# Train a regression model
python saluki.py mode=fine-tune task=regression \
    data_source.filepath=my_training_data.txt \
    data_source.splitratio=[80,10,10] \
    training.learningrate=0.001

# Train with a file that has a header row
python saluki.py mode=fine-tune task=regression \
    data_source.filepath=my_data_with_header.txt \
    data_source.stripheader=true \
    data_source.splitratio=[80,10,10]

# Make predictions (assuming model was saved to 'my_experiment/fine-tune/0')
python saluki.py mode=predict task=regression \
    data_source.filepath=my_test_data.txt \
    inference.pretrainedmodel=my_experiment/fine-tune/0

# Generate interpretations
python saluki.py mode=interpret task=regression \
    data_source.filepath=my_test_data.txt \
    inference.pretrainedmodel=my_experiment/fine-tune/0
```

## Data Format

Your input data should be a tab-separated file with the following columns:

```text
sequence_id    sequence_data    label
ENST000001    A,U,G,C,A,G...    1.23
ENST000002    C,G,A,U,G,C...    0.89
ENST000003    U,A,C,G,U,A...    2.45
```

- **sequence_id**: Unique identifier for each sequence (string)
- **sequence_data**: RNA sequence (comma-separated nucleotides: A,C,G,U)
- **label**: Numeric value for regression (float) or class label for classification (string - will be auto-encoded to integers)

**Example file content (save as tab-separated .txt file):**

```text
ENST000001    A,U,G,C,A,G    1.234
ENST000002    C,G,A,U,G,C    0.567
ENST000003    U,A,C,G,U,A    class_A
```

**Important Notes:**

- No header row in the data file (unless `data_source.stripheader=true`)
- Sequences can be of variable length
- For classification, use any string labels (e.g., "class_A", "class_B") - they will be automatically converted to integers
- File should be saved with `.txt` extension

## Command Line Interface

The framework uses Hydra for configuration management. All parameters can be set via command line:

```bash
python saluki.py [mode=MODE] [task=TASK] [parameter=value]...
```

### Available Modes

- `tokenize`: Learn sequence tokenization
- `fine-tune`: Train models on labeled data
- `predict`: Make predictions with trained models
- `interpret`: Generate LOO scores for interpretability

### Available Tasks

- `regression`: For continuous numeric predictions
- `classification`: For categorical predictions

### Getting Help

```bash
# General help
python saluki.py --help

# Mode-specific help
python saluki.py mode=fine-tune --help

# Task-specific help
python saluki.py mode=fine-tune task=regression --help
```

## Advanced Configuration

### Using Config Files

For complex setups, create YAML config files and override specific parameters:

```bash
python saluki.py --config-path configs --config-name my_config \
    data_source.filepath=new_data.txt \
    training.learningrate=0.0005
```

### Cross-Validation

```bash
# 5-fold cross-validation (note: splitratio is train/validation only for CV)
python saluki.py mode=fine-tune task=regression \
    data_source.filepath=data.txt \
    data_source.crossvalidation=5 \
    data_source.splitratio=[80,20]
```

### Custom Data Splits

If your data includes predefined split columns:

```bash
python saluki.py mode=fine-tune task=regression \
    data_source.filepath=data_with_splits.txt \
    data_source.splitpos=4 \
    data_source.devsplits=[1,2] \
    data_source.testsplits=[3]
```

## Advanced Data Preprocessing

The framework supports advanced preprocessing options via the `settings.data_pre_processing` section:

```bash
# Center sequences around a specific token (e.g., CDS start)
python saluki.py mode=fine-tune task=regression \
    settings.data_pre_processing.centertoken="ATG" \
    data_source.filepath=data.txt

# Filter to only sequences of specific lengths
python saluki.py mode=fine-tune task=regression \
    settings.data_pre_processing.only512=true \
    data_source.filepath=data.txt

# Extract specific sequence regions
python saluki.py mode=fine-tune task=regression \
    settings.data_pre_processing._3utr=true \
    data_source.filepath=data.txt
```

Available preprocessing options:

- `centertoken`: Center sequences around a specific token
- `only512`: Filter to only 512-token sequences
- `_3utr`: Extract 3' UTR regions
- `non3utr`: Extract non-3' UTR regions
- `nomarkers`: Remove special marker tokens

## Output Structure

Results are saved in the `outputpath` directory:

```text
experiment_output/
├── tokenizer.json              # Trained tokenizer
├── tokenize/
│   └── logs/                   # Tokenization logs
├── fine-tune/
│   ├── 0/                      # Cross-validation fold 0
│   │   ├── pytorch_model.bin   # Model weights
│   │   ├── test_predictions.csv # Predictions
│   │   ├── all_results.json    # Metrics
│   │   └── trainer_state.json  # Training state
│   └── tboard/                 # TensorBoard logs
├── predictions/
│   ├── test_predictions.csv    # Model predictions
│   ├── rank_deltas.csv        # Performance metrics
│   └── logs/                  # Execution logs
└── interpretations/
    ├── loo_scores_remove.csv  # LOO scores (CSV)
    ├── loo_scores_remove.pkl  # LOO scores (SHAP format)
    └── logs/                  # Execution logs
```

## Key Parameters

### Data Configuration

- `data_source.filepath`: Path to your data file
- `data_source.stripheader`: Whether to skip the first line (header) in the data file (default: false)
- `data_source.columnsep`: Column separator in data file (default: "\t")
- `data_source.tokensep`: Token separator within sequences (default: ",")
- `data_source.splitratio`: Train/val/test split ratios (e.g., [80,10,10])
- `data_source.crossvalidation`: Number of CV folds
- `data_source.splitpos`: Column index for predefined splits

### Training Parameters

- `training.learningrate`: Learning rate (default: 0.001)
- `training.batchsize`: Batch size for training
- `training.nepochs`: Number of training epochs
- `training.patience`: Early stopping patience
- `training.seed`: Random seed for reproducibility (default: 42)
- `training.scaling`: Label scaling method ("log", "minmax", "standard", or "none")
- `training.resume`: Whether to resume training from checkpoint (default: false)

### Tokenization Parameters

- `tokenization.encoding`: Encoding type ("atomic" or "bpe")
- `tokenization.samplesize`: Number of samples to use for tokenizer training
- `tokenization.lefttailing`: Whether to truncate from left when sequences are too long

## Architecture

Built on:

- **PyTorch**: Deep learning framework
- **Transformers**: Model architecture library
- **Hydra**: Configuration management
- **BioLM Utils**: RNA-specific utilities

## Citation

If you use this software, please cite the original Saluki paper:

```text
Saluki: alignment-free estimation of amino acid substitution rates
Authors et al.
Genome Biology, 2022
```

## Troubleshooting

### Common Issues

**CUDA out of memory**: Reduce batch size with `training.batchsize=4`

**Invalid sequence format**: Ensure sequences are comma-separated nucleotides (A,C,G,U)

**Config not found**: Use absolute paths or check working directory

### Debugging Options

For development and troubleshooting, use these debugging parameters:

```bash
# Use only a subset of data for quick testing
python saluki.py mode=fine-tune task=regression \
    debugging.dev=100 \
    data_source.filepath=data.txt

# Run in silent mode (less verbose output)
python saluki.py mode=fine-tune task=regression \
    debugging.silent=true \
    data_source.filepath=data.txt

# Force reload data even if cached
python saluki.py mode=fine-tune task=regression \
    debugging.forcenewdata=true \
    data_source.filepath=data.txt

# Specify number of GPUs to use (auto-detected by default)
python saluki.py mode=fine-tune task=regression \
    debugging.ngpus=2 \
    data_source.filepath=data.txt
```

**Debugging Parameters:**

- `debugging.dev`: Use only N samples for quick testing (default: false)
- `debugging.silent`: Reduce logging verbosity (default: false)
- `debugging.forcenewdata`: Force dataset recreation even if cached (default: false)
- `debugging.ngpus`: Number of GPUs to use (auto-detected, default: 1)
- `debugging.accelerator`: Hardware accelerator ("gpu" or "cpu", default: "gpu")

### Getting Support

- Check the [BioLM Utils documentation](https://github.com/dieterich-lab/biolm_utils)
- Open an issue on the [RNA Saluki CNN repository](https://github.com/dieterich-lab/rna_saluki_cnn) for bugs or feature requests
