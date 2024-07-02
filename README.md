# `rna_protein_saluki`: A plugin to run bioinformatical CNN-RNN Models.

This projects implements fine-tuning of the [Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) for regressing half lives of RNA and protein sequences. In addition, it supports the extraction of leave-one-out (LOO) scores for fine-tuned models to analyse importance scores of individual inputs.

In detail, the following steps are implemented:

- Tokenization of RNA/Protein sequences via one-hot encoding of molecules.
- Fine-tune models for regression.
- Calculation of leave-one-out scores for you fine-tuned model.

## Installation

First clone the repo and cd into it. Then, we recommend to create a dedicated environment ([python venv](https://docs.python.org/3/library/venv.html)) for the project. Now, you install the project via the [Pipfile](./Pipfile) file which in turn will install the [biolm_utils](https://github.com/dieterich-lab/biolm_utils) library. Summarising, excute the following steps:

```bash
git clone https://github.com/dieterich-lab/rna_protein_saluki.git
cd rna_protein_saluki
python3 -m venv biolm_saluki 
. biolm/biolm_saluki/activate
pip install pipenv
pipenv install
```

## File structure

```bash
├── exampleconfigs # exampleconfigs to work with
├── Pipfile # installation file
├── README.md
├── rna_saluki.py # Implementation of the `RNABaseDataset`, espcially implementing the `__getitem__()` method.
├── rna_saluki.py # Implementation of the models, espcially implementing the `getconfig()` method.
├── saluki.py # Main script importing the `run()` function from `biolm_utils` and declaration of the model/data/training configuration.
```

### Usage

The main script is [saluki.py](./saluki.py) which import the `run()` function from the [biolm._utils](https://github.com/dieterich-lab/biolm_utils) library. It will access the given parameters from the parameters in [`params.py`](biolm_utils/params.py) and additionally from a custom `Config` object.py](/biolm_utils/config.py) that can be set via `set_config()`. The script can be run via

```bash
python saluki.py [tokenize | pre-train | fine-tune | predict | interepret]
```

To get a verbose exlplanation on all the possible parameters you can run the following:

```bash
python saluki.py -h 
```

For general usage we refer user to the [README](https://github.com/dieterich-lab/biolm_utils/blob/main/README.md) of the `biolm_utils` framework.
