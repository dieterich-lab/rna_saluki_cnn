from __future__ import annotations

from typing import Any

import torch.nn as nn
from biolm.base_dataset import BaseModel

# Saluki model constants
SALUKI_BLOCKSIZE = 12288

# The full implementation lives in `HFSaluki` below. We expose `SalukiModel`
# as an alias to the huggingface-compatible implementation so the framework
# can reference `SalukiModel` directly from plugin imports.


class MidBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_width,
        maxpool_width,
        maxpool_stride,
        dropout,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_width,
        )
        self.layernorm = nn.LayerNorm(normalized_shape=output_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_width, stride=maxpool_stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.relu(x)
        return x


class EntryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_width
        )
        self.layernorm = nn.LayerNorm(normalized_shape=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, channels, dropout, out_features):
        super().__init__()
        self.l1 = nn.Linear(in_features=channels, out_features=channels)
        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm = nn.BatchNorm1d(num_features=channels)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(in_features=channels, out_features=out_features)

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class GruBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            batch_first=True,
            bidirectional=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, x = self.gru(x)
        x = x.squeeze(0)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Saluki(nn.Module):
    def __init__(
        self,
        input_size,
        channels=64,
        kernel_width=5,
        maxpool_width=2,
        maxpool_stride=2,
        layers=6,
        num_labels=1,
        dropout=0.3,
    ):
        super().__init__()
        self.entry_block = EntryBlock(
            in_channels=input_size, out_channels=channels, kernel_width=kernel_width
        )
        self.mid_blocks = nn.ModuleList(
            [
                MidBlock(
                    channels,
                    channels,
                    kernel_width,
                    maxpool_width,
                    maxpool_stride,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.gru_block = GruBlock(channels=channels)
        self.dense_block = DenseBlock(
            channels=channels, dropout=dropout, out_features=num_labels
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.entry_block(x)
        for block in self.mid_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = x.flip(1)
        x = self.gru_block(x)
        x = self.dense_block(x)
        return x


class HFSaluki(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Accept optional architecture parameters from config (useful for testing with small sequences)
        self.model = Saluki(
            input_size=config.input_size,
            num_labels=config.num_labels,
            channels=getattr(config, 'hidden_size', 64),
            kernel_width=getattr(config, 'conv_kernel_size', 5),
            maxpool_width=getattr(config, 'pool_size', 2),
            maxpool_stride=getattr(config, 'pool_size', 2),
            layers=getattr(config, 'num_layers', 6),
            dropout=getattr(config, 'dropout', 0.3),
        )

    def forward(self, input_ids, **kwargs):
        x = self.model(input_ids)
        return {"logits": x}

    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        from transformers import PretrainedConfig

        # Saluki invariants
        # Access blocksize from training config
        blocksize = (
            getattr(args.training, "blocksize", None)
            if hasattr(args, "training")
            else None
        )
        if blocksize is None:
            blocksize = getattr(args, "blocksize", None)

        if blocksize not in (None, SALUKI_BLOCKSIZE):
            raise ValueError(
                f"Saluki requires blocksize={SALUKI_BLOCKSIZE}; do not set a different blocksize in global config."
            )

        # Check encoding
        tokenization = getattr(args, "tokenization", None)
        if tokenization and getattr(tokenization, "encoding", None) != "atomic":
            raise ValueError(
                f"Saluki requires encoding='atomic', but got '{getattr(tokenization, 'encoding', None)}'. "
                "Saluki requires atomic encoding."
            )

        if config_cls is not None:
            config = config_cls(
                vocab_size=len(tokenizer),
                max_position_embeddings=SALUKI_BLOCKSIZE,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            # For custom models not using HF config, use PretrainedConfig
            config = PretrainedConfig(
                vocab_size=len(tokenizer),
                max_position_embeddings=SALUKI_BLOCKSIZE,
                pad_token_id=tokenizer.pad_token_id,
            )
        cat0 = dataset.OHE.categories_[0]
        try:
            cat_size = cat0.size
        except Exception:
            cat_size = len(cat0)
        config.input_size = cat_size + dataset.nspecs
        config.num_labels = nlabels
        return config


# Expose the full model implementation as `SalukiModel` for the framework.
SalukiModel = HFSaluki
