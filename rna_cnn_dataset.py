import numpy as np
import torch

from biolm_utils.rna_datasets import RNABaseDataset


class RNACNNDataset(RNABaseDataset):
    def __getitem__(self, i):
        example = self.examples[i].copy()
        example["input_ids"] = self.OHE.transform(
            np.reshape(example["input_ids"], (-1, 1))
        )
        if self.args.specifiersep is not None:
            spec = self.specs[i]
            example["input_ids"] = np.concatenate((example["input_ids"], spec), axis=1)
        example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.float)
        return example
