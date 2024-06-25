from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast
from transformers.image_processing_utils import BaseImageProcessor

from biolm_utils.config import Config, set_config
from models import HFSaluki
from rna_cnn_dataset import RNACNNDataset

params = [
    HFSaluki,  # 0
    PreTrainedTokenizerFast,  # 1
    RNACNNDataset,  # 2
    1e-5,  # 3
    1.0,  # 4
    0.0,  # 5
    BaseImageProcessor,  # 6
    None,  # 6
    DefaultDataCollator,  # 8
    False,  # 9
    BertConfig,  # 10
    False,  # 11
]

config = Config(*params)
set_config(config)

from biolm_utils.biolm import run

run()
