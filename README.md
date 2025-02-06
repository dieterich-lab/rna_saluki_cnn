# `rna_saluki_cnn`: A plugin to run bioinformatical CNN-RNN Models.

This projects implements fine-tuning of the [Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) model for the tasks of `regression` and `classification`. In addition, it supports the extraction of leave-one-out (LOO) scores for fine-tuned models to analyse importance scores of individual inputs.

In detail, the following steps are implemented:

- Tokenization of RNA sequences via one-hot encoding of molecules.
- Fine-tune models for regression and classification.
- Calculation of leave-one-out scores for you fine-tuned model.

## Installation

First clone the repo and cd into it. Then, we recommend to create a dedicated environment ([python venv](https://docs.python.org/3/library/venv.html)) for the project. Now, you install the project via the [Pipfile](./Pipfile) file which in turn will install the [biolm_utils](https://github.com/dieterich-lab/biolm_utils) library. Summarising, execute the following steps:

```bash
git clone --recurse-submodules https://github.com/dieterich-lab/rna_saluki_cnn.git
cd rna_saluki_cnn
python3 -m venv ~/.venvs/biolm_saluki  # or any other choice of directory
. ~/.venvs/biolm_saluki/bin/activate # or your choice of directory
pip install pipenv
pipenv install
```

To update this repository execute:

```bash
git pull
```

To also update the `biolm_utils` framework, type in:

```bash
git pull --recurse-submodules
```


## File structure

```bash
├── exampleconfigs # exampleconfigs to work with
├── Pipfile # installation file
├── README.md
├── rna_cnn_models.py # Implementation of the model, espcially implementing the `getconfig()` method.
├── saluki.py # Main script importing the `run()` function from `biolm_utils` and declaration of the model/data/training configuration.
```

## Usage

The main script is [saluki.py](./saluki.py) which imports the `run()` function from the [biolm._utils](https://github.com/dieterich-lab/biolm_utils) library and provides the a custom `Config` object suitable for running the saluki model. The script can be run via

```bash
python saluki.py {tokenize, fine-tune, predict, interpret} {regression, classification}
```

To get a verbose exlplanation on all the possible parameters for each mode you can run the following:

```bash
python saluki.py {tokenize, fine-tune, predict, interpret} -h 
```

Please adhere to the [example workflow](#example-workflow) to retrace the single steps. For specific usage and information about the configuration parameters we refer the user to the [command line options section](#command-line-options).

## Example config files


We offer two example config files. The first one is for the pipeline of **tokenization** and **fine-tuning**. The other one is for **predicting** (inference on a test file) and **interpreting** (generation of LOO scores). The latter one is noticeably smaller as all the training cofigurations fall away.

```bash
exampleconfigs
├── predict_interpret.yaml
├── tokenize_fine-tune.yaml
```

## Pathing and Results

The software will save all experiment data in the [`outputpath`](#0-designate-an-outputpath) (or fall back to the file path stem of the input file if not given). This directory will be created if not existant. There, we will save the dataset (tokenized samples from the given filepath), the tokenizer and the models. 

### Tokenizing (`tokenize`) and fine-tuning (`fine-tune`):

Assuming we use cross valdiation via 3 splits and having fine-tuned a model, the directory will look as follows (commented are only files concerning your results):
```bash

├── fine-tune
│   ├── 0
│   │   ├── all_results.json # combined results for training, evalution & test
│   │   ├── checkpoint-xxx 
│   │   ├── config.json
│   │   ├── eval_results.json # evaluation results
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin
│   │   ├── rank_deltas.csv # a file showing the rank deltas (i.e. for calculating spearman correlation) for the test data
│   │   ├── special_tokens_map.json 
│   │   ├── test_predictions.csv # the predictions of the model on the test set
│   │   ├── test_results.json # test results 
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   ├── trainer_state.json
│   │   ├── training_args.bin
│   │   └── train_results.json # training loss
│   ├── 1
│   │   ├── ... # same as in "0"
│   ├── 2
│   │   ├── ... # same as in "0"
│   └── tboard
│       ├── events.out.tfevents.x.gpu.x # tensorboard runs, showing loss, learning rate and so on
├── tokenize
│   └── logs
│       ├── YY-MM-hh:mm.log # the log files of the your run (or multiple runs)
└── tokenizer.json
```

### Inference (`predict`) and interpration (`interpret`):

Assuming we would use the directory "predictions for `predict` and "looscores" for `interpret`, the the results in the directories will look as follows:

```bash
predictions
├── dataset.json  # saved dataset for quicker load when run multiple times (can be deleted)
├── logs # log folder
├── rank_deltas.csv # file denoting the spearman rank for each sample
└── test_predictions.csv # file denoting the prediction for each sample
```

```bash
looscores
├── dataset.json  # saved dataset for quicker load when run multiple times (can be deleted)
├── logs # log folder
├── loo_scores_replace.csv # the .csv file containing the results (in this case for each replacment). Header is `sequence,token,replacement,label,pred,start_offset,end_offset,loo`
└── loo_scores_replace.pkl # same as above, but as `shap.Explanation` object for easier analysis with the `shap` library.
```

The header of the `loo_scores_{handletokens}.csv` can be read as follows:
- `sequence`: The sequence id / identifier
- `token`: the actual token (for `remove` it was deleted from the sequence, for `mask` it's one-hot encoding was set to zero, for `replace` it was replaced with the token under `replacement`, see below)
- `replacement`: Only valid for `handletokens: replace`, see above
- `label`: The true regression value / class
- `pred`: The predicted regression value / class
- `start_offset`: Start offset in the sequence (zero-indexed)
- `end_offset`: End offset in the sequence (zero-indexed). Example: The `a` in `cgat` would have start/end index of (2, 3)
- `loo`: The loo score: positive means, the prediction increased for the value of `loo`, negative means, the predictions decreased for that amount

## Example workflow

This tutorial will lead you through an end-to-end process of training a tokenizer and fine-tuning a model. When you have questions about the arguments used here, you can read in detail about them in the [command line options](#command-line-options) section of this README.

### 0) Designate an Outputpath

First off and simple, you can provide a path where to save your experiments (see [Pathing and Results](#pathing-and-results)):

```yaml
#
# Below is the experimentname; an identifier that will make your experiment re-usable.
#
outputpath: experiments/rna_saluki  # If None, will be set to the file name (without extension)
```

### 1) Data Configuration

We designed options to give varying data sources for either tokenzation and for the fine-tuning step (if you are using the same file, just mirror the parameters accordingly). You also have to let the scripts know where exactly to find information about labels, sequences and splits in your data file. The two according sections in the config file are listed below. Attributes should be self-explanatory by their comments or explained by the command line parser. (see [usage](#usage)). Don't be confused by the mention of "pre-training"; this corresponds to the parser of the `biolm_utils` framework, but this plugin will not make use of it.

```yaml
#
# Description of the datasource used for 
# - training the tokenizer 
# - pre-training (for LM)
#
tokenizing and pre-training data source:
  filepath: "tokenizing_data_file.txt" # this is the path to the file that you use to learn the tokenizer.
  stripheader: False # if this data file has a header that has to be stripped.
  columnsep: "\t" # could be [",", "|", "\t", ...] This denominates field separator.
  tokensep: "," # This denominates how input tokens are concatenated (use "" or `False` if your input sequence is a conesecutive string of tokens).
  specifiersep: None # If you modified your input tokens, this denominates the separator token (see below for further explanations).
  idpos: 1 # Position of the identifier column of your data, e.g. "ENST00000488147", which will be printed out in the inference/interepret results.
  seqpos: 1 # Position of the actual sequence in your file (your "input data").
  ```

Once again, if your fine-tuning data is the one you learned the tokenizer from, please mirror the entries from above to the below segment in the yaml file.

```yaml
#
# Description of the fine-tuning source
#
fine-tuning data source:
  filepath: "fine-tuning_data_file.txt" # this is the path to the file that you use to learn the tokenizer.
  stripheader: False # if the custom data file has a header that has to be stripped.
  columnsep: "\t" # could be [",", "|", "\t", ...] This denominates field separator.
  tokensep: "," # This denominates how input tokens are concatenated (use "" or `False` if your input sequence is a conesecutive string of tokens).
  specifiersep: None # If you modified your input tokens, this denominates the separator token (see below for further explanations).
  idpos: 1 # Position of the identifier column of your data, e.g. "ENST00000488147", which will be printed out in the inference/interepret results.
  seqpos: 1 # Position of the actual sequence in your file (your "input data").
  labelpos: 1 # Position of the label column.
  weightpos: None # Position of the column containing quality labels with allowed labels: ["STRONG", "GOOD", "WEAK", "POOR"].
  splitpos: 1 # If your data contains designated splits (at least 3) for which we can carry out cross validation. If there is not such a column, just change to `False` (see below for further explanation).
  splitratio:  # If your data has no designated splits, you can denote a comma-separated list as split ratio like `80,20` or `70,20,10`. If that list contains a third split, testing is triggered on that split, otherwise, no testing is done.
  ```

A prototypical dataset file would look like this (without header)

```csv
0	ENST00000488147	ENSG00000227232	653635	WASH7P	unprocessed_pseudogene	0.204213162843933	3.39423360819142	0.121582579281952	0.374739086478062	a,t,g,g,g,a,g,c,c,g,t,g,t,g,c,a,c,g,t,c,g,g,g,a,g,c,t,c,g,g,a,g,t,g,a,g,c,gej,c,a,c,c,a,t,g,a,c,t,c,c,t,g,t,g,a,g,g,a,t,g,c,a,g,c,a,c,t,c,c,c,t,g,g,c,a,g,g,t,c,a,g,a,c,c,t,a,t,g,c,c,g,t,g,c,c,c,t,t,c,a,t,c,c,a,g,c,c,a,g,a,c,c,t,g,c,g,g,c,g,a,g,a,g,g,a,g,g,c,c,g,t,c,c,a,g,c,a,g,a,t,g,g,c,g,g,a,t,g,c,c,c,t,g,c,a,g,t,a,c,c,t,g,c,a,g,a,a,g,g,t,c,t,c,t,g,g,a,g,a,c,a,t,c,t,t,c,a,g,c,a,g,gej,t,a,g,a,g,c,a,g,a,g,c,c,g,g,a,g,c,c,a,g,g,t,g,c,a,g,g,c,c,a,t,t,g,g,a,g,a,g,a,a,g,g,t,c,t,c,c,t,t,g,g,c,c,c,a,g,g,c,c,a,a,g,a,t,t,g,a,g,a,a,g,a,t,c,a,a,g,g,g,c,a,g,c,a,a,g,a,a,g,g,c,c,a,t,c,a,a,g,gej,t,g,t,t,c,t,c,c,a,g,t,g,c,c,a,a,g,t,a,c,c,c,t,g,c,t,c,c,a,g,g,g,c,g,c,c,t,g,c,a,g,g,a,a,t,a,t,g,g,c,t,c,c,a,t,c,t,t,c,a,c,g,g,g,c,g,c,c,c,a,g,g,a,c,c,c,t,g,g,c,c,t,g,c,a,g,a,g,a,c,g,c,c,c,c,c,g,c,c,a,c,a,g,g,a,t,c,c,a,g,a,g,c,a,a,g,c,a,c,c,g,c,c,c,c,c,t,g,g,a,c,g,a,g,c,g,g,g,c,c,c,t,g,c,a,g,gej,a,g,a,a,g,c,t,g,a,a,g,g,a,c,t,t,t,c,c,t,g,t,g,t,g,c,g,t,g,a,g,c,a,c,c,a,a,g,c,c,g,g,a,g,c,c,c,g,a,g,g,a,c,g,a,t,g,c,a,g,a,a,g,a,g,g,g,a,c,t,t,g,g,g,g,g,t,c,t,t,c,c,c,a,g,c,a,a,c,a,t,c,a,g,c,t,c,t,g,t,c,a,g,c,t,c,c,t,t,g,c,t,g,c,t,c,t,t,c,a,a,c,a,c,c,a,c,c,g,a,g,a,a,c,c,t,gej,t,a,g,a,a,g,a,a,g,t,a,t,g,t,c,t,t,c,c,t,g,g,a,c,c,c,c,c,t,g,g,c,t,g,g,t,g,c,t,g,t,a,a,c,a,a,a,g,a,c,c,c,a,t,g,t,g,a,t,g,c,t,g,g,g,g,g,c,a,g,a,g,a,c,a,g,a,g,g,a,g,a,a,g,c,t,g,t,t,t,g,a,t,g,c,c,c,c,c,t,t,g,t,c,c,a,t,c,a,g,c,a,a,g,a,g,a,g,a,g,c,a,g,c,t,g,g,a,a,c,a,g,c,a,g,gej,t,c,c,c,a,g,a,g,a,a,c,t,a,c,t,t,c,t,a,t,g,t,g,c,c,a,g,a,c,c,t,g,g,g,c,c,a,g,g,t,g,c,c,t,g,a,g,a,t,t,g,a,t,g,t,t,c,c,a,t,c,c,t,a,c,c,t,g,c,c,t,g,a,c,c,t,g,c,c,c,g,g,c,a,t,t,g,c,c,a,a,c,g,a,c,c,t,c,a,t,g,t,a,c,a,t,t,g,c,c,g,a,c,c,t,g,g,g,c,c,c,c,g,g,c,a,t,t,g,c,c,c,c,c,t,c,t,g,c,c,c,c,t,g,g,c,a,c,c,a,t,t,c,c,a,g,a,a,c,t,g,c,c,c,a,c,c,t,t,c,c,a,c,a,c,t,g,a,g,g,t,a,g,c,c,g,a,g,c,c,t,c,t,c,a,a,g,aej,c,c,t,a,c,a,a,g,a,t,g,g,g,g,t,a,c,t,a,a,c,a,c,c,a,c,c,c,c,c,a,c,c,g,c,c,c,c,c,a,c,c,a,c,c,a,c,c,c,c,c,a,g,c,t,c,c,t,g,a,g,g,t,g,c,t,g,g,c,c,a,g,t,g,c,a,c,c,c,c,c,a,c,t,c,c,c,a,c,c,c,t,c,a,a,c,c,g,c,g,g,c,c,c,c,t,g,t,a,g,g,c,c,a,a,g,g,c,g,c,c,a,g,g,c,a,g,g,a,c,g,a,c,a,g,c,a,g,c,a,g,c,a,g,c,g,c,g,t,c,t,c,c,t,t,c,a,g,tej,c,c,a,g,g,g,a,g,c,t,c,c,c,a,g,g,g,a,a,g,t,g,g,t,t,g,a,c,c,c,c,t,c,c,g,g,t,g,g,c,t,g,g,c,c,a,c,t,c,t,g,c,t,a,g,a,g,t,c,c,a,t,c,c,g,c,c,a,a,g,c,t,g,g,g,g,g,c,a,t,c,g,g,c,a,a,g,g,c,c,a,a,g,c,t,g,c,g,c,a,g,c,a,t,g,a,a,g,g,a,g,c,g,a,a,a,g,c,t,g,g,a,g,a,a,g,c,a,g,c,a,g,c,a,g,a,a,g,g,a,g,c,a,g,g,a,g,c,a,a,g,tej,g,a,g,a,g,c,c,a,c,g,a,g,c,c,a,a,g,g,t,g,g,g,c,a,c,t,t,g,a,t,g,t,c,gej,c,t,c,c,a,t,g,g,g,g,g,g,a,c,g,g,c,t,c,c,a,c,c,c,a,g,c,c,t,g,c,g,c,c,a,c,t,g,t,g,t,t,c,t,t,a,a,g,a,g,g,c,t,t,c,c,a,g,a,g,a,a,a,a,c,g,g,c,a,c,a,c,c,a,a,t,c,a,a,t,a,a,a,g,a,a,c,t,g,a,g,c,a,g,a,a,a
```

> **Attention**:There are certain specifics regarding the following entries:
> - `splitpos`: If it is set to `None` fine-tuning is carried out on the splits denoted in `splitratio`. If a splits position is given, we expect at least three different splits on which we do cross validation by:
>  - setting each split as a dedicated test set
>  - setting the succesor split as a dedicated validation set
>  - and training on the rest of the splits.
> - `splitratio`:Comma-seprated list describing the desired split ratio for train, validation and (possibly) test split in the format `train_percentage/val_percentage(/test_percentage)`, e.g. `85,15` or `70,20,10`. Must sum up to 100 (see default). Given a third splitratio triggers testing on that split.
>- `specifiersep`: If you want to decorate your atomic tokens with float numbers you can do so, by denoting a separator after which you append the float number(s) to the atomic token. For example, you could specify `specifiersep: #` for generating your samples as: `a#2.5, c, A, g#5.7, ...` or even with multiple modiefiers like `a#2.5#0.2, c, A, g#5.7, ...` . The decorating float numbers are then appended to new "channels" of the one-hot encoding. Regarding the last sample from above, this would result in a one-hot-encoding of (assuming a vocabulary of `[a, c, g, t, A, C, G, T]`):

```
a | 1  | 0 | 0 | 0 |
c | 0  | 1 | 0 | 0 |
g | 0  | 0 | 0 | 1 |
t | 0  | 0 | 0 | 0 |
A | 0  | 0 | 1 | 0 |
C | 0  | 0 | 0 | 0 |
G | 0  | 0 | 0 | 0 |
T | 0  | 0 | 0 | 0 |
  |2.5 | 0 | 0 |5.7|
  |0.2 | 0 | 0 |5.7|
```

- `weightpos` (regression only): We can carry out weighted regression by weighting the loss of labels with quality labels of `["STRONG", "GOOD", "WEAK", "POOR"]` with correpsonding weights of `[0.25, 0.5, 0.75, 1]`.

### 2) Tokenization

The term "tokenization" originates from language modelling terminolgy and originally refers to splitting a contiguous sequence into subparts (tokens) and "learning a tokenizer" usually involves some statistical processes like byte pair encoding. But for this project we will simply split the sequences into individual atomic characters (see the example under [data configuration](#1-data-configuration)). These characters are then mapped to one-hot-encodings (and optionally modificaton channels).

To train a tokenizer, you'll be using the `tokenize` mode (here no `--task` flag is necessary):

```bash
python saluki.py tokenize --configfile exampleconfigs/tokenize_fine-tune.yaml
```
The only option is to downsample your file for "learning" a tokenizer if it is huge, albeit this option is rather important for other realisations of the `biolm_utils` framework.

> **Attention**: Do not change the `encoding` as this is the default encoding of one-hot-encodings for CNN inputs.

```bash
#
# If you want to tokenize, you only need to specify the following.
#
tokenization:
  samplesize: None # if your data file is to big to learn a tokenizer, you can downsample it
  encoding: atomic # DO NOT CHANGE. This is the default encoding of one-hot-encodings for CNN inputs.
  lefttailing: True # If true, the sequences will be cut from the left (begging from the right end).
```


### 3) Fine-tuning a model

For fine-tuning (training) a model, the `fine-tune` mode together with the `--task` flag of choice (`{classification, regression}`) is required, e.g.

```bash
python saluki.py fine-tune --task regression --configfile exampleconfigs/tokenize_fine-tune.yaml
```

for regression and 

```bash
python saluki.py fine-tune --task classification --configfile exampleconfigs/tokenize_fine-tune.yaml
```

for classification tasks.

Depending on the `splitpos` argument, fine-tuning will be carried out on a 90/10 train/eval split or via cross validaton on each split as validation set. In your config file you can make certain modifications to the training settings:

> **Attention**: Do not change the `blocksize` as this is the default sequence length for the CNN-RNN to function properly.

```yaml
settings:
  data pre-processing:
    centertoken: False # either False or a character on which the sequence will be centered
  environment:
    ngpus: 1 # [1, 2, 4] # under development: automatically infer this from the environment
  training:
    general:
      batchsize: 8 # This is the batch size. (effective gradients will be batchsize x gradacc, see below)
      gradacc: 4 # Gradient accumulation: Determines how many batches of gradients should be aggregated (effective gradients will be batchsize x gradacc)
      blocksize: 12288 # DO NOT CHANGE. This is the default sequence length for the CNN-RNN to work.
      nepochs: 10 # Number of epochs the model iterates over the training dataset.
      patience: 3 # Number of evaluation (once per epoch) that are carried out without improvements of the model on the evaluation set before training is stopped.
      resume: False # When a training was cancelled (resuming) or further fine-tuning (see the general documentation of biolm_utils for further details.
      scaling: log # label scaling [log, minmax, standard]
      weightedregression: False # if you have quality labels for your data, then you can do weighted regression. Please fill out "weightpos" under "fine-tuning data source".
```

### 4) Inference (predicting)

Now that you've trained a model (new models) you probably want to make predictions on new data. To do so, you can use `predict` mode with the `--task` flag of choice (`{classification, regression}`): 

```bash
python saluki.py predict --task {regression, classification} --configfile exampleconfigs/predict_interpret.yaml
```

As a lot of the training parameters are obsolete for pure inference, we provide a [slimmer inference config file](exampleconfigs/predict_interpret.yaml) for this purpose. It's now all about declaring the structure of the new data source, where to save the results and where to find the trained model to infer from. The latter will point  to a folder, where all the model specific files are stored (like `pytorch_model.bin` and so on, see [Pathing and Results](#pathing-and-results)):

```yaml
outputpath: "test_folder"  # If None, will be set to the file name (without extension)

inference data source:
  filepath: "data_to_be_predicted_or_to_be_inferred_from.txt"
  stripheader: False # if the custom data file has a header that has to be stripped
  columnsep: "\t" # could be "," "|", "\t" ...
  tokensep: ","
  specifiersep: None
  idpos: 1 # position of the identifier of the column 
  seqpos: 2 # position of the sequence column 
  labelpos: 3 # if the file has ground truth labels, this is the position of the label column (else delete or set to `None`)

#
# State the encoding of the pretrained model
#
tokenization:
  encoding: atomic # DO NOT CHANGE. This is the default encoding of one-hot-encodings for CNN inputs.

inference model:
  pretrainedmodel: "path/to/fine-tuned-model" # path of the fine-tuned model to infer from

#
# Genral settings for model predictons.
#
settings:
  data pre-processing:
    centertoken: False # either False or a character on which the sequence will be centered
  environment:
    ngpus: 1 # [1, 2, 4] # TODO: automatically infer this from the environment
  training:
    batchsize: 8
    blocksize: 12288 # DO NOT CHANGE. This is the default sequence length for the CNN-RNN to work.
    scaling: log # label scaling [log, minmax, standard]
```

### 5) Interpretation

As a last step, you certainly want to get intepretations for your predictions.  To do so, you can use `interpret` mode with the `--task` flag of choice (`{classification, regression}`): 

```bash
python saluki.py interpret --task {regression, classification} --configfile exampleconfigs/predict_interpret.yaml
```

Similar to [inference](#4-inference-predicting), most of the training parameters are obsolete, so we provide a [slimmer inference config file](exampleconfigs/predict_interpret.yaml). For Interpretability, we resort to [leave-one-out scores](https://aclanthology.org/N19-1357.pdf). "Leaving out" a token can be handled in three different ways:

- `remove`: The token will be completely removed from the sequence.
- `mask`: The token will be replaced with the tokenizer's `[MASK]` token.
- `replace`: The token will be exchanged for against other tokens specified by `replacementslist`. In the example below, `a` is replaced against `[b, c]`, `b` against `[a, c]` and so on.

As for inference, in the config file you should declare the new data source, where to save the results and where to find the trained model to infer from. 

> **Attention**: Although the calculation of LOO scores is batched, it is still fairly expensive:
>
>    - For `remove`/`mask`: In a sequence of 1,000 tokens each token will either be removed or replaced its one-hot-vector set to zero which results in 1,000 samples for single sequence.
>    - For `replace`: In a sequence of 1,000 tokens each token will be replaced by X mutual tokens, resulting in 1,000 * X samples.


```yaml
outputpath: "test_folder"  # If None, will be set to the file name (without extension)

inference data source:
  filepath: "data_to_be_predicted_or_to_be_inferred_from.txt"
  stripheader: False # if the custom data file has a header that has to be stripped
  columnsep: "\t" # could be "," "|", "\t" ...
  tokensep: ","
  specifiersep: None
  idpos: 1 # position of the identifier of the column 
  seqpos: 2 # position of the sequence column 
  labelpos: 3 # if the file has ground truth labels, this is the position of the label column (else delete or set to `None`)

#
# State the encoding of the pretrained model
#
tokenization:
  encoding: atomic # DO NOT CHANGE. This is the default encoding of one-hot-encodings for CNN inputs.

inference model:
  pretrainedmodel: "path/to/fine-tuned-model" # path of the fine-tuned model to infer from

#
# Genral settings for model predictons.
#
settings:
  data pre-processing:
    centertoken: False # either False or a character on which the sequence will be centered
  environment:
    ngpus: 1 # [1, 2, 4] # TODO: automatically infer this from the environment
  training:
    batchsize: 8
    blocksize: 12288 # DO NOT CHANGE. This is the default sequence length for the CNN-RNN to work.
    scaling: log # label scaling [log, minmax, standard]

#
# Interpretation settings
#
looscores:
  handletokens: remove # One of [remove, mask, replace]. This determines how to treat the absence of a token during leave-one-out calculation.
  replacementlists: [["a", "b", "c"], ["x", "y", "z"]] # List of lists of atomic tokens that should be replaced against each other if `handletokens` is set to `replace`.
  replacespecifier: True # if `True` and `handletokens` is set to `replace`, modified tokens (i.e. "a#0.7") will also be relplaced against an unmodified version (e.g. "a#0.7" --> ["c#0.7", "g#0.7", "t#0.7", "a"])`.
```