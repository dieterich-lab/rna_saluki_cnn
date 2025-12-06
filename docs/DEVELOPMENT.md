# Saluki Plugin Development Guide

Guide for developing and extending the Saluki plugin for BioLM 2.0.

## 🎯 Overview

The Saluki plugin implements a CNN-based model for RNA sequence analysis. This guide covers:

- Plugin architecture and structure
- Development workflow
- Testing strategies
- Adding new features
- Debugging tips

---

## 🏗️ Plugin Architecture

### Plugin Structure

```
saluki_plugin/
├── __init__.py              # Package initialization
├── config.py                # Plugin configuration (entry point)
├── rna_cnn_dataset.py       # RNACNNDataset class
└── rna_cnn_models.py        # HFSaluki model
```

### Entry Point Registration

The plugin is discovered via entry point in `pyproject.toml`:

```toml
[tool.poetry.plugins."biolm.plugins"]
saluki = "saluki_plugin.config:get_config"
```

**How it works:**
1. BioLM scans `biolm.plugins` entry point group
2. Finds `saluki` entry pointing to `get_config()` function
3. Calls `get_config()` to get plugin configuration
4. Uses returned `PluginConfig` to instantiate models/datasets

---

## 📦 Plugin Configuration

### `saluki_plugin/config.py`

The `get_config()` function returns a `PluginConfig` object:

```python
from biolm.plugin_config import PluginConfig
from saluki_plugin.rna_cnn_models import HFSaluki
from saluki_plugin.rna_cnn_dataset import RNACNNDataset
from transformers import PreTrainedTokenizerFast, DefaultDataCollator

def get_config() -> PluginConfig:
    """Saluki plugin configuration."""
    return PluginConfig(
        # Saluki doesn't support pre-training
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=HFSaluki,
        
        # Dataset and tokenizer
        dataset_cls=RNACNNDataset,
        tokenizer_cls=PreTrainedTokenizerFast,
        
        # Data collators
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        
        # Configuration
        add_special_tokens=False,  # Atomic encoding
        pretraining_required=False,  # No pre-training
    )
```

**Key points:**
- `model_cls_for_pretraining=None` - Saluki doesn't support pre-training
- `add_special_tokens=False` - Uses atomic nucleotide encoding
- `pretraining_required=False` - Can fine-tune directly

---

## 🧬 Model Implementation

### HFSaluki CNN Architecture

Located in `saluki_plugin/rna_cnn_models.py`:

```python
class HFSaluki(PreTrainedModel):
    """CNN-based model for RNA sequence analysis."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Embedding layer (nucleotide → vector)
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.hidden_size if i == 0 else config.num_filters,
                out_channels=config.num_filters,
                kernel_size=config.kernel_size,
                padding='same'
            )
            for i in range(config.num_layers)
        ])
        
        # Classification/regression head
        self.classifier = nn.Linear(
            config.num_filters,
            config.num_labels
        )
    
    def forward(self, input_ids, labels=None):
        # input_ids: [batch_size, seq_len]
        # Embed nucleotides
        x = self.embeddings(input_ids)  # [batch, seq, hidden]
        
        # Transpose for Conv1d: [batch, hidden, seq]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Global max pooling
        x = x.max(dim=-1)[0]  # [batch, num_filters]
        
        # Predict
        logits = self.classifier(x)  # [batch, num_labels]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss() if config.num_labels == 1 else nn.CrossEntropyLoss()
            loss = loss_fn(logits.squeeze(), labels)
        
        return {"loss": loss, "logits": logits}
```

**Key design choices:**
- **Atomic encoding:** Each nucleotide is a single token
- **Conv1d layers:** Capture local sequence motifs
- **Global max pooling:** Position-invariant aggregation
- **Simple head:** Linear layer for prediction

---

## 📊 Dataset Implementation

### RNACNNDataset

Located in `saluki_plugin/rna_cnn_dataset.py`:

```python
class RNACNNDataset(Dataset):
    """Dataset for RNA sequences with comma-separated nucleotides."""
    
    def __init__(self, filepath, tokenizer, column_ids=[1, 2, 3]):
        """
        Args:
            filepath: Path to TSV file
            tokenizer: Tokenizer for encoding
            column_ids: [id_col, label_col, sequence_col] (1-indexed)
        """
        self.data = []
        
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split('\t')
                
                # Extract columns (convert to 0-indexed)
                seq_id = parts[column_ids[0] - 1]
                label = float(parts[column_ids[1] - 1])
                sequence = parts[column_ids[2] - 1]
                
                # Tokenize: "a,t,g,c" → [1, 2, 3, 4]
                tokens = tokenizer.encode(sequence)
                
                self.data.append({
                    'input_ids': tokens,
                    'labels': label,
                    'seq_id': seq_id
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

**Important:**
- Expects comma-separated nucleotides: `a,t,g,c,a,g,t,c,...`
- Column IDs are **1-indexed** (user-friendly)
- Returns dict with `input_ids`, `labels`, `seq_id`

---

## 🧪 Testing

### Test Structure

```
tests/
├── test_saluki_full_pipeline.py      # End-to-end tests
└── test_saluki_plugin_config.py      # Configuration tests
```

### Running Tests

```bash
# All tests
poetry run pytest tests/

# Specific file
poetry run pytest tests/test_saluki_full_pipeline.py -v

# With coverage
poetry run pytest tests/ --cov=saluki_plugin --cov-report=html
```

### Writing New Tests

```python
def test_my_feature():
    """Test description."""
    # Arrange
    config = get_config()
    model = config.model_cls_for_finetuning()
    
    # Act
    result = model(input_ids)
    
    # Assert
    assert result is not None
```

---

## 🔧 Development Workflow

### 1. Setup Development Environment

```bash
# Clone repo
git clone https://github.com/dieterich-lab/rna_saluki_cnn.git
cd rna_saluki_cnn

# Install with dev dependencies
poetry install --with dev

# Verify tests pass
poetry run pytest tests/
```

### 2. Make Changes

```bash
# Create feature branch
git checkout -b feature/my-feature

# Edit code
vim saluki_plugin/rna_cnn_models.py

# Run tests
poetry run pytest tests/ -v
```

### 3. Code Quality

```bash
# Check style
ruff check saluki_plugin/

# Format code
ruff format saluki_plugin/

# Type checking (optional)
mypy saluki_plugin/
```

### 4. Commit and Push

```bash
git add saluki_plugin/
git commit -m "Add my feature"
git push origin feature/my-feature
```

---

## 🎨 Adding New Features

### Example: Add Layer Normalization

1. **Update model:**

```python
# saluki_plugin/rna_cnn_models.py
class HFSaluki(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # ... existing code ...
        
        # Add layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.num_filters)
            for _ in range(config.num_layers)
        ])
    
    def forward(self, input_ids, labels=None):
        # ... existing code ...
        
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            x = conv(x)
            x = norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm on features
            x = F.relu(x)
        
        # ... rest of forward ...
```

2. **Add test:**

```python
# tests/test_saluki_plugin_config.py
def test_layer_normalization():
    """Test that layer normalization is applied."""
    config = get_config()
    model = config.model_cls_for_finetuning()
    
    # Check layer norms exist
    assert hasattr(model, 'layer_norms')
    assert len(model.layer_norms) == model.config.num_layers
```

3. **Update documentation:**

```markdown
## Model Architecture

- Layer normalization after each convolution
- Improves training stability
```

---

## 🐛 Debugging Tips

### Common Issues

**1. Plugin not discovered:**

```bash
# Check entry point
poetry run python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='biolm.plugins')
print([ep.name for ep in eps])
"

# Should show: ['saluki', ...]
```

**Fix:** Reinstall plugin: `cd rna_saluki_cnn && poetry install`

---

**2. Import errors:**

```python
# Error: ModuleNotFoundError: No module named 'saluki_plugin'

# Check installation
poetry run python -c "import saluki_plugin; print(saluki_plugin.__file__)"

# Should show path to installed package
```

**Fix:** Install in development mode: `poetry install`

---

**3. Model dimension mismatch:**

```python
# Error: RuntimeError: size mismatch, m1: [8 x 256], m2: [512 x 1]

# Check model forward
print(f"After conv: {x.shape}")
print(f"After pooling: {x.shape}")
print(f"Classifier expects: {model.classifier.in_features}")
```

**Fix:** Ensure pooling output matches classifier input

---

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now see detailed logs
config = get_config()
model = config.model_cls_for_finetuning()
```

---

## 📚 Best Practices

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Keep functions focused and small

### Testing

- Test happy paths and edge cases
- Mock expensive operations (file I/O, training)
- Use fixtures for common setup
- Aim for >80% coverage

### Documentation

- Update README for user-facing changes
- Add docstrings to new classes/functions
- Include examples in docstrings
- Keep this dev guide updated

---

## 🔗 Related Resources

- **[BioLM Plugin Development](https://github.com/dieterich-lab/biolm_utils/blob/biolm-2.0/docs/PLUGIN_DEVELOPMENT.md)** - Framework guide
- **[PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)** - CNN basics
- **[HuggingFace PreTrainedModel](https://huggingface.co/docs/transformers/main_classes/model)** - Model base class

---

## 🚀 Next Steps

- [ ] Add attention mechanism
- [ ] Support variable-length sequences
- [ ] Add more test coverage
- [ ] Optimize for longer sequences (>12K)
- [ ] Add visualization tools

---

**Questions?** Open an issue on [GitHub](https://github.com/dieterich-lab/rna_saluki_cnn/issues)
