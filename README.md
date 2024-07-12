# `rna_protein_saluki`: A plugin to run bioinformatical CNN-RNN Models.

This projects implements fine-tuning of the [Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) model for regressing half lives of RNA and protein sequences. In addition, it supports the extraction of leave-one-out (LOO) scores for fine-tuned models to analyse importance scores of individual inputs.

In detail, the following steps are implemented:

- Tokenization of RNA/Protein sequences via one-hot encoding of molecules.
- Fine-tune models for regression.
- Calculation of leave-one-out scores for you fine-tuned model.

## Installation

First clone the repo and cd into it. Then, we recommend to create a dedicated environment ([python venv](https://docs.python.org/3/library/venv.html)) for the project. Now, you install the project via the [Pipfile](./Pipfile) file which in turn will install the [biolm_utils](https://github.com/dieterich-lab/biolm_utils) library. Summarising, excute the following steps:

```bash
git clone https://github.com/dieterich-lab/rna_saluki_cnn.git
cd rna_saluki_cnn
python3 -m venv ~/.venvs/biolm_saluki  # or any other choice of directory
. ~/.venvs/biolm_saluki/bin/activate # or your choice of directory
pip install pipenv
pipenv install
```

## File structure

```bash
├── exampleconfigs # exampleconfigs to work with
├── Pipfile # installation file
├── README.md
├── rna_cnn_models.py # Implementation of the models, espcially implementing the `getconfig()` method.
├── saluki.py # Main script importing the `run()` function from `biolm_utils` and declaration of the model/data/training configuration.
```

## Usage

The main script is [saluki.py](./saluki.py) which imports the `run()` function from the [biolm._utils](https://github.com/dieterich-lab/biolm_utils) library and provides the a custom `Config` object suitable for running the saluki model. The script can be run via

```bash
python saluki.py [tokenize | fine-tune | predict | interepret]
```

To get a verbose exlplanation on all the possible parameters you can run the following:

```bash
python saluki.py -h 
```

Please adhere to the [example workflow](#example-workflow) to retrace the single steps. For specific usage and information about the configuration parameters we refer the user to the [README](https://github.com/dieterich-lab/biolm_utils/blob/main/README.md) of the `biolm_utils` framework. 

## Dummy config files

We offer two dummy config files. The first one is for the pipeline of **tokenization**, **fine-tuning**, **testing** (testing is also done during fine-tuning, but can be also again separately invoked) and **extracting loo scores**. The other one is for **inference** (getting predictions on new files) and **interpret** modes. The latter one is noticeably smaller as all the training cofigurations fall away.

```bash
exampleconfigs
├── inference_interpret.yaml
├── tokenize_fine_tune_test_interpret.yaml
```

## Example workflow

### 1) Data Configuration

We designed options to give varying data sources for either tokenzation and for the fine-tuning step (if you are using the same file, just mirror the parameters accordingly). You also have to let the scripts know where exactly to find information about labels, sequences and splits in your data file. The two according sections in the config file are listed below. Attributes should be self-explanatory by their comments or explained by the command line parser. (see [usage](#usage)). Don't be confused by the mention of "pre-training"; this corresponds to the parser of the `biolm_utils` framework, but this plugin will not make use of it.

```yaml
#
# Description of the datasource used for 
# - training the tokenizer 
# - pre-training (for LM)
#
tokenizing and pre-training data source:
  filepath: "tokenizing_and_pre-training_data_file.txt"
  stripheader: False # if the custom data file has a header that has to be stripped
  columnsep: "\t" # could be "," "|", "\t" ...
  tokensep: ","
  specifiersep: None
  idpos: 1 # position of the identifier of the column 
  seqpos: 1 # position of the sequence column 
  pretrainedmodel: None # if the tokenizer for pre-training diverts from the chosen data.

#
# Description of the fine-tuning source
#
fine-tuning data source:
  filepath: "fine-tuning_data_file.txt"
  stripheader: False # if the custom data file has a header that has to be stripped
  columnsep: "\t" # could be "," "|", "\t" ...
  tokensep: ","
  specifiersep: None
  idpos: 1 # position of the identifier of the column 
  seqpos: 1 # position of the sequence column 
  labelpos: 1 # position of the label column 
  weightpos: None # position of the column containing quality labels 
  splitpos: 1 # position of the split identifier for cross validaton
  pretrainedmodel: None # if the pre-trained model diverts from the chosen data.
  ```

An example prototypical dataset file would look like this (without header)

```csv
0	ENST00000488147	ENSG00000227232	653635	WASH7P	unprocessed_pseudogene	0.204213162843933	3.39423360819142	0.121582579281952	0.374739086478062	a,t,g,g,g,a,g,c,c,g,t,g,t,g,c,a,c,g,t,c,g,g,g,a,g,c,t,c,g,g,a,g,t,g,a,g,c,gej,c,a,c,c,a,t,g,a,c,t,c,c,t,g,t,g,a,g,g,a,t,g,c,a,g,c,a,c,t,c,c,c,t,g,g,c,a,g,g,t,c,a,g,a,c,c,t,a,t,g,c,c,g,t,g,c,c,c,t,t,c,a,t,c,c,a,g,c,c,a,g,a,c,c,t,g,c,g,g,c,g,a,g,a,g,g,a,g,g,c,c,g,t,c,c,a,g,c,a,g,a,t,g,g,c,g,g,a,t,g,c,c,c,t,g,c,a,g,t,a,c,c,t,g,c,a,g,a,a,g,g,t,c,t,c,t,g,g,a,g,a,c,a,t,c,t,t,c,a,g,c,a,g,gej,t,a,g,a,g,c,a,g,a,g,c,c,g,g,a,g,c,c,a,g,g,t,g,c,a,g,g,c,c,a,t,t,g,g,a,g,a,g,a,a,g,g,t,c,t,c,c,t,t,g,g,c,c,c,a,g,g,c,c,a,a,g,a,t,t,g,a,g,a,a,g,a,t,c,a,a,g,g,g,c,a,g,c,a,a,g,a,a,g,g,c,c,a,t,c,a,a,g,gej,t,g,t,t,c,t,c,c,a,g,t,g,c,c,a,a,g,t,a,c,c,c,t,g,c,t,c,c,a,g,g,g,c,g,c,c,t,g,c,a,g,g,a,a,t,a,t,g,g,c,t,c,c,a,t,c,t,t,c,a,c,g,g,g,c,g,c,c,c,a,g,g,a,c,c,c,t,g,g,c,c,t,g,c,a,g,a,g,a,c,g,c,c,c,c,c,g,c,c,a,c,a,g,g,a,t,c,c,a,g,a,g,c,a,a,g,c,a,c,c,g,c,c,c,c,c,t,g,g,a,c,g,a,g,c,g,g,g,c,c,c,t,g,c,a,g,gej,a,g,a,a,g,c,t,g,a,a,g,g,a,c,t,t,t,c,c,t,g,t,g,t,g,c,g,t,g,a,g,c,a,c,c,a,a,g,c,c,g,g,a,g,c,c,c,g,a,g,g,a,c,g,a,t,g,c,a,g,a,a,g,a,g,g,g,a,c,t,t,g,g,g,g,g,t,c,t,t,c,c,c,a,g,c,a,a,c,a,t,c,a,g,c,t,c,t,g,t,c,a,g,c,t,c,c,t,t,g,c,t,g,c,t,c,t,t,c,a,a,c,a,c,c,a,c,c,g,a,g,a,a,c,c,t,gej,t,a,g,a,a,g,a,a,g,t,a,t,g,t,c,t,t,c,c,t,g,g,a,c,c,c,c,c,t,g,g,c,t,g,g,t,g,c,t,g,t,a,a,c,a,a,a,g,a,c,c,c,a,t,g,t,g,a,t,g,c,t,g,g,g,g,g,c,a,g,a,g,a,c,a,g,a,g,g,a,g,a,a,g,c,t,g,t,t,t,g,a,t,g,c,c,c,c,c,t,t,g,t,c,c,a,t,c,a,g,c,a,a,g,a,g,a,g,a,g,c,a,g,c,t,g,g,a,a,c,a,g,c,a,g,gej,t,c,c,c,a,g,a,g,a,a,c,t,a,c,t,t,c,t,a,t,g,t,g,c,c,a,g,a,c,c,t,g,g,g,c,c,a,g,g,t,g,c,c,t,g,a,g,a,t,t,g,a,t,g,t,t,c,c,a,t,c,c,t,a,c,c,t,g,c,c,t,g,a,c,c,t,g,c,c,c,g,g,c,a,t,t,g,c,c,a,a,c,g,a,c,c,t,c,a,t,g,t,a,c,a,t,t,g,c,c,g,a,c,c,t,g,g,g,c,c,c,c,g,g,c,a,t,t,g,c,c,c,c,c,t,c,t,g,c,c,c,c,t,g,g,c,a,c,c,a,t,t,c,c,a,g,a,a,c,t,g,c,c,c,a,c,c,t,t,c,c,a,c,a,c,t,g,a,g,g,t,a,g,c,c,g,a,g,c,c,t,c,t,c,a,a,g,aej,c,c,t,a,c,a,a,g,a,t,g,g,g,g,t,a,c,t,a,a,c,a,c,c,a,c,c,c,c,c,a,c,c,g,c,c,c,c,c,a,c,c,a,c,c,a,c,c,c,c,c,a,g,c,t,c,c,t,g,a,g,g,t,g,c,t,g,g,c,c,a,g,t,g,c,a,c,c,c,c,c,a,c,t,c,c,c,a,c,c,c,t,c,a,a,c,c,g,c,g,g,c,c,c,c,t,g,t,a,g,g,c,c,a,a,g,g,c,g,c,c,a,g,g,c,a,g,g,a,c,g,a,c,a,g,c,a,g,c,a,g,c,a,g,c,g,c,g,t,c,t,c,c,t,t,c,a,g,tej,c,c,a,g,g,g,a,g,c,t,c,c,c,a,g,g,g,a,a,g,t,g,g,t,t,g,a,c,c,c,c,t,c,c,g,g,t,g,g,c,t,g,g,c,c,a,c,t,c,t,g,c,t,a,g,a,g,t,c,c,a,t,c,c,g,c,c,a,a,g,c,t,g,g,g,g,g,c,a,t,c,g,g,c,a,a,g,g,c,c,a,a,g,c,t,g,c,g,c,a,g,c,a,t,g,a,a,g,g,a,g,c,g,a,a,a,g,c,t,g,g,a,g,a,a,g,c,a,g,c,a,g,c,a,g,a,a,g,g,a,g,c,a,g,g,a,g,c,a,a,g,tej,g,a,g,a,g,c,c,a,c,g,a,g,c,c,a,a,g,g,t,g,g,g,c,a,c,t,t,g,a,t,g,t,c,gej,c,t,c,c,a,t,g,g,g,g,g,g,a,c,g,g,c,t,c,c,a,c,c,c,a,g,c,c,t,g,c,g,c,c,a,c,t,g,t,g,t,t,c,t,t,a,a,g,a,g,g,c,t,t,c,c,a,g,a,g,a,a,a,a,c,g,g,c,a,c,a,c,c,a,a,t,c,a,a,t,a,a,a,g,a,a,c,t,g,a,g,c,a,g,a,a,a
```

There are certain specifics regarding the following entries:

- `splitpos`: If it is set to `None` fine-tuning is carried out on a 90/10 train/val split with no subsequent testing. If a splits position is given, we expect at least three different splits on which we do cross validation by:
  - setting each split as a dedicated test set
  - setting the following split as a dedicated validation set
  - and training on the rest of the splits.

- `specifiersep` (**one-hot encoding only**): If you want to decorate your atomic tokens with float numbers you can do so, by denoting a separator after which you append the float number(s) to the atomic token. For example, you could specify `specifiersep: #` for generating your samples as: `a#2.5, c, A, g#5.7, ...` or even with multiple modiefiers like `a#2.5#0.2, c, A, g#5.7, ...` . The decorating float numbers are then appended to new "channels" of the one-hot encoding. Regarding the last sample from above, this would result in a one-hot-encoding of (assuming a vocabulary of `[a, c, g, t, A, C, G, T]`):

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

### 2) Tokenization

The term "tokenization" originates form language modelling terminolgy and originally referes to splitting a contiguous sequence into subparts (tokens). For this project we will split the sequences into individual atomic characters which (see the example under [data configuration](#1-data-configuration)). The whole process maps characters to one-hot-encodings (and optionally modificaton channels)

To train a tokenizer, you'll beusing the `tokenize` mode:

```bash
python saluki.py tokenize --configfile exampleconfigs/tokenize_fine-tune_test_interpret.yaml
```
There are no real parameters to change in the configfile. The only valid option is to downsample your file for "learning" a tokenizer if it is huge, albeit this option is rather important for other realisations of the `biolm_utils` framework.

> **Attention**: Do not change the `eencoding` as this is the default encoding of one-hot-encodings for CNN inputs.

```bash
#
# If you want to tokenize, you only need to specify the following.
#
tokenization:
  samplesize: None # if your data file is to big to learn a tokenizer, you can downsample it
  encoding: atomic # DO NOT CHANGE. This is the default encoding of one-hot-encodings for CNN inputs.
```


### 3) Fine-tuning a model

For fine-tuning (training) a model, the `fine-tune` mode is required:

```bash
python saluki.py fine-tune --configfile exampleconfigs/tokenize_fine-tune_test_interpret.yaml
```

In your config file you can make certain modifications to the training settings:

> **Attention**: Do not change the `blocksize` as this is the default sequence length for the CNN-RNN to function properly.

```yaml
settings:
  data pre-processing:
    centertoken: False # either False or a character on which the sequence will be centered
  environment:
    ngpus: 1 # [1, 2, 4] # under development: automatically infer this from the environment
  training:
    general:
      batchsize: 8
      gradacc: 4 # Gradient accumulation (effective gradients will be batchsize x gradacc)
      blocksize: 12888 # DO NOT CHANGE. This is the default sequence length for the CNN-RNN to work.
      nepochs: 10
      patience: 3
      resume: False # for resuming training
      scaling: log # label scaling [log, minmax, standard]
      weightedregression: False # if you have quality labels for your data, then you can do weighted regression. Please fill out "weightpos" under "fine-tuning data source".
```