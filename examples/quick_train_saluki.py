"""Quick smoke demo: programmatic tiny training run using Saluki plugin + biolm_utils.

This script demonstrates the minimal, programmatic flow described in the
biolm_utils docs. It registers/discovers plugins, builds a tiny dataset and
model, uses the framework's trainer helper, and runs a single epoch.

Intended to be a short local sanity check for developer environments.
"""

import sys
from pathlib import Path

# When this script is executed directly (python examples/quick_train_saluki.py)
# sys.path[0] will be the examples/ directory, so the local project root
# (which contains the `biolm_utils` package) won't be on the import path.
# Add the repository root to sys.path so imports still work when running the
# script file directly.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from biolm_utils.plugin_loader import discover_entrypoint_plugins

# Ensure entry-points are discovered (if the plugin was installed)
discover_entrypoint_plugins()

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DefaultDataCollator, TrainingArguments

from biolm_utils.structured_config import BioLMConfig, DebuggingConfig, TrainingConfig
from biolm_utils.train_utils import compute_metrics_for_regression, get_trainer
from biolm_utils.trainer import RegressionTrainer


class TinyDataset(Dataset):
    def __init__(self, n=8):
        self.items = [
            (torch.randn(1, 10), torch.tensor(float(i % 2))) for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return {"input_ids": x, "labels": y}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 1)

    def forward(self, input_ids, **kwargs):
        x = input_ids.squeeze(1) if len(input_ids.shape) > 2 else input_ids
        return {"logits": self.lin(x.float())}


def main():
    args = BioLMConfig(
        mode="fine-tune",
        task="regression",
        debugging=DebuggingConfig(dev=False, silent=True),
        training=TrainingConfig(patience=1, batchsize=2),
    )

    train_ds = TinyDataset(8)
    val_ds = TinyDataset(4)

    model = TinyModel()

    targs = TrainingArguments(
        output_dir=str(Path("/tmp") / "saluki_demo_outputs"),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        disable_tqdm=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )
    targs.label_names = ["labels"]

    # Simple compute metrics wrapper
    def compute_metrics(pred):
        fn = compute_metrics_for_regression(val_ds, Path("/tmp"))
        return fn((pred.predictions, pred.label_ids))

    trainer = get_trainer(
        args,
        RegressionTrainer,
        model,
        None,
        targs,
        train_ds,
        val_ds,
        DefaultDataCollator(),
        compute_metrics,
        None,
    )
    print("Starting tiny training run (1 epoch)...")
    trainer.train()
    print("Done. Trainer state:", getattr(trainer, "state", None))


if __name__ == "__main__":
    main()
