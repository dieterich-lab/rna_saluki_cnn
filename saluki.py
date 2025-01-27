from biolm_utils.config import Config, set_config
from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast
from transformers.image_processing_utils import BaseImageProcessor

from rna_cnn_dataset import RNACNNDataset
from rna_cnn_models import HFSaluki

params = [
    None,
    HFSaluki,
    PreTrainedTokenizerFast,
    1e-3,
    0.4,
    0.001,
    BaseImageProcessor,
    None,
    DefaultDataCollator,
    False,
    BertConfig,
    False,
    RNACNNDataset,
]

config = Config(*params)
set_config(config)

from biolm_utils.biolm import run

run()
