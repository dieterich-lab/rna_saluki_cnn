from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast
from transformers.image_processing_utils import BaseImageProcessor

from models import HFSaluki
from rna_cnn_dataset import RNACNNDataset

# Collate models.
MODELCLS = HFSaluki
# Collate tokenizers.
TOKENIZER_CLS = PreTrainedTokenizerFast

DATASET_CLS = RNACNNDataset

LEARNINGRATE = 1e-5

MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0

SPECIAL_TOKENIZER_FOR_TRAINER_CLS = BaseImageProcessor
DATACOLLATOR_CLS_FOR_PRETRAINING = None
DATACOLLATOR_CLS_FOR_FINETUNING = DefaultDataCollator

ADD_SPECIAL_TOKENS = False

CONFIGCLS = BertConfig

PRETRAINING_REQUIRED = False
