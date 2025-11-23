from __future__ import annotations

from typing import Any, List

from torch.utils.data import Dataset


class SalukiDataset(Dataset):
    def __init__(self, items: List[Any] | None = None):
        self._items = items or [0, 1, 2]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]
