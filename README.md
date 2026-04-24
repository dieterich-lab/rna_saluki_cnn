> **Note:** The `saluki-2.0` branch contains the latest, actively developed version of this plugin. The `main` branch is legacy. For the newest features and code, please [switch to the `saluki-2.0` branch](https://github.com/dieterich-lab/rna_saluki_cnn/tree/saluki-2.0).

# Saluki Plugin for BioLM 2.0

**CNN-based RNA regulatory prediction plugin**

This plugin integrates Saluki, a convolutional neural network, into the BioLM framework for:

- **RNA sequence tokenization**
- **Supervised learning** (classification/regression)
- **Feature interpretation**

## Installation

1. Install the BioLM framework:

   ```bash
   git clone https://github.com/dieterich-lab/biolm_utils.git
   cd biolm_utils
   git checkout biolm-2.0
   ./install.sh
   ```

2. Install the Saluki plugin from the `saluki-2.0` branch:

   ```bash
   poetry run biolm install-plugin "https://github.com/dieterich-lab/rna_saluki_cnn.git?ref=saluki-2.0"
   ```

## Developer install

When making changes to the Saluki plugin, install it directly from your local checkout so Python uses the editable source:

```bash
poetry run biolm develop-plugin /path/to/rna_saluki_cnn
```

## Usage

After installation, Saluki integrates seamlessly with the BioLM framework. Since Saluki is a CNN-based model, it **does not require pre-training** but **does require tokenization** (atomic encoding of nucleotides).

### Quick Start Configuration

Create a configuration file (e.g., `config.yaml`) for your experiment:

```yaml
plugin: saluki
outputpath: /tmp/saluki_experiment
task: classification  # or 'regression'
data_source:
  filepath: /path/to/your/data.tsv
  columnsep: "\t"
  idpos: 1
  seqpos: 2
  labelpos: 3
  splitratio: [80, 10, 10]
training:
  nepochs: 10
  batchsize: 8
```

### Training Commands

**Tokenize your data (required):**

```bash
poetry run biolm mode=tokenize plugin=saluki data_source.filepath=/path/to/data.tsv outputpath=/tmp/saluki_run
```

**Fine-tune on your tokenized data:**

```bash
poetry run biolm mode=fine-tune plugin=saluki task=classification data_source.filepath=/path/to/data.tsv outputpath=/tmp/saluki_run
```

**Make predictions:**

```bash
poetry run biolm mode=predict plugin=saluki task=classification data_source.filepath=/path/to/test_data.tsv inference.pretrainedmodel=/path/to/model.safetensors outputpath=/tmp/saluki_run
```

**Interpret features:**

```bash
poetry run biolm mode=interpret plugin=saluki task=classification data_source.filepath=/path/to/data.tsv inference.pretrainedmodel=/path/to/model.safetensors outputpath=/tmp/saluki_run
```

### Data Format

Saluki expects tab-separated data with columns for ID, label, and sequence:

```tsv
ID	Label	Sequence
seq_001	1.5	AUGCUAGCUAGC
seq_002	2.3	AUGGCUAUGGCU
```

- `idpos`: Column index (1-based) for sequence IDs
- `seqpos`: Column index for RNA sequences
- `labelpos`: Column index for labels (numeric for regression, 0/1 for classification)

### Configuration Options

Key configuration parameters for Saluki:

- **Task**: `classification` or `regression`
- **Training**: `nepochs`, `batchsize`, `learning_rate` (default: 0.001)
- **Data**: `splitratio` for train/validation/test splits
- **Interpretation**: `inference.looscores.handletokens` (`mask` or `remove`)

For complete configuration options, see the [BioLM configuration documentation](https://github.com/dieterich-lab/biolm_utils/blob/biolm-2.0/README.md#configuration-management).

## Preprocessing Pipeline

The `tools/preprocessing_pipeline.py` script is a single-command utility designed to extract and enrich transcripts and save them in a TXT file which can directly be used as an input to the Saluki model. It uses `gffread` to extract sequences based on a reference genome and a GTF annotation file, and then processes the transcripts to add biological markers (like Exon Junctions and CDS boundaries). Optionally, it can also merge the final data with additional metadata from a CSV using a YAML configuration file.

### Usage

```bash
python tools/preprocessing_pipeline.py genome_fasta annotation_gtf output_dir [options]
```

### Positional Arguments
* **`genome_fasta`**: Path to the toplevel reference genome FASTA file (e.g., `Homo_sapiens.GRCh38.dna.toplevel.fa` found at https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/).
* **`annotation_gtf`**: Path to the corresponding gene annotation GTF file (e.g., `Homo_sapiens.GRCh38.115.gtf` found at https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/).
* **`output_dir`**: Directory where all intermediate and final output files (SQLite DB, extracted FASTA, and final TXT) will be stored.

### Optional Arguments
* **`--create_db`**: Flag to force the creation of a new GTF SQLite database. If omitted, the script will use the existing DB if it is already available in the output directory.
* **`--output_txt`**: Name of the final enriched transcripts output file. (Default: `enriched_transcripts.txt`)
* **`--ej_markers`**: Whether to add exon junction markers (`ej` or `EJ`) into the sequence. Accepts standard booleans or strings like 'true', '1', 'yes'. (Default: `True`)
* **`--cds_markers`**: Whether to mark CDS sequences. If enabled, the starting bases of codons are marked with uppercase letters, while UTRs and non-starting bases are lowercase. (Default: `True`)
* **`--merge_yaml`**: Path to a YAML configuration file to merge additional columns from an external CSV into the final text output. The YAML file must contain: `csv_path`, `columns_to_append`, `txt_id_col` (1-based index), and `csv_id_col` (1-based index).

  **Example `merge_config.yaml`:**
  ```yaml
  csv_path: "/path/to/data/half_life.csv"
  columns_to_append: ["half_life", "rate", "rate.min", "rate.max"] # Column names (or 1-based indices) to extract
  txt_id_col: 1  # 1-based index of the column containing the join ID in the generated TXT
  csv_id_col: 1  # 1-based index of the column containing the join ID in the target CSV
  ```

## Output Files
Running the script generates the following files in the specified `output_dir`:
1. **SQLite Database** (`<gtf_basename>.db`): Used for fast querying of exons, CDS, and stop codon features.
2. **Extracted FASTA** (`extracted_transcripts.fa`): The raw sequence file generated automatically by invoking `gffread`.
3. **Enriched Transcripts** (`enriched_transcripts.txt`): The final tab-separated dataset containing transcript context (Transcript ID, Gene ID, HGNC Symbol, Biotype) and the stringently formatted marker sequences.
