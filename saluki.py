from biolm_utils.config import Config, set_config
from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast
from transformers.image_processing_utils import BaseImageProcessor

from models import HFSaluki
from rna_cnn_dataset import RNACNNDataset

params = [
    None,  # 0
    HFSaluki,  # 1
    PreTrainedTokenizerFast,  # 2
    RNACNNDataset,  # 3
    1e-3,  # 4
    0.4,  # 5
    0.001,  # 6
    BaseImageProcessor,  # 7
    None,  # 8
    DefaultDataCollator,  # 9
    False,  # 10
    BertConfig,  # 11
    False,  # 12
]

config = Config(*params)
set_config(config)

from biolm_utils.biolm import run

run()
