import json
import logging
import re
import tempfile

import numpy as np
import pandas as pd
import transformers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from biolm_utils.train_utils import IdentityScaler, LogScaler


class RNABaseDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        add_special_tokens,
    ):
        self.tokenizer = tokenizer
        self.args = args
        # Prepare helpers and resolved attributes for structured config
        data_source = getattr(args, "data_source", None)
        tokenization = getattr(args, "tokenization", None)
        settings = getattr(args, "settings", None)

        def ds_get(key, default=None):
            if data_source is not None and hasattr(data_source, key):
                return getattr(data_source, key)
            return getattr(args, key, default)

        def tk_get(key, default=None):
            if tokenization is not None and hasattr(tokenization, key):
                return getattr(tokenization, key)
            return getattr(args, key, default)

        def settings_get(key, default=None):
            if settings is not None:
                dp = getattr(settings, "data_pre_processing", None)
                if isinstance(dp, dict) and key in dp:
                    return dp.get(key)
            return getattr(args, key, default)

        self.nspecs = 0
        self.specs = None
        self.OHE = None
        if getattr(args, "task", None) == "classification":
            self.LE = LabelEncoder()

        # Resolve filepath (prefer nested data_source.filepath)
        filepath = getattr(getattr(args, "data_source", None), "filepath", None)
        if not filepath:
            filepath = getattr(args, "filepath", None)

        with open(filepath, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
            if ds_get("stripheader", False):
                lines = lines[1:]

        # We'll save the original input data lines for later reference.
        self.lines = lines
        columnsep = ds_get("columnsep", "\t")
        idpos = ds_get("idpos", None)
        self.seq_idx = [x.split(columnsep)[idpos - 1].strip('"') for x in self.lines]

        tokensep = ds_get("tokensep", None)
        encoding = tk_get("encoding", "atomic")
        self.join_str = "" if tokensep is None or encoding == "bpe" else tokensep

        # Expose frequently-used config options on the instance for use in
        # helper methods (backwards compatible with legacy flat top-level attributes).
        self.encoding = encoding
        self.centertoken = settings_get(
            "centertoken", getattr(args, "centertoken", None)
        )
        self.only512 = settings_get("only512", getattr(args, "only512", False))
        self._3utr = settings_get("_3utr", getattr(args, "_3utr", False))
        self.non3utr = settings_get("non3utr", getattr(args, "non3utr", False))
        self.nomarkers = settings_get("nomarkers", getattr(args, "nomarkers", False))

        # Normalize and pre-trokenize to obtain the sequences.
        normalized_seqs = [
            tokenizer.backend_tokenizer.normalizer.normalize_str(x) for x in lines
        ]
        # Keep a copy of normalized lines for helper methods that expect them
        self.normalized_lines = normalized_seqs

        logging.info("Normalizing sequences finished.")

        specifiersep = ds_get("specifiersep", None)
        if specifiersep is not None:
            with open(tokenizer.name_or_path, "r") as f:
                tokenizer_json = json.load(f)
            tokenizer_json["normalizer"]["normalizers"].pop(-3)
            tokenizer_json["pre_tokenizer"]["pretokenizers"].pop(-1)
            with tempfile.NamedTemporaryFile("r+") as tmp:
                json.dump(tokenizer_json, tmp)
                tmp.seek(0)
                spec_tokenizer = tokenizer.__class__(
                    tokenizer_file=tmp.name,
                    mask_token="[MASK]",
                    cls_token="[CLS]",
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                    sep_token="[SEP]",
                    bos_token="[BOS]",
                    eos_token="[EOS]",
                    model_max_length=getattr(
                        getattr(args, "training", None),
                        "blocksize",
                        getattr(args, "blocksize", None),
                    ),
                    truncation=True,
                    truncation_side=(
                        "left" if tk_get("lefttailing", False) else "right"
                    ),
                )
            spec_normalized_seqs = [
                spec_tokenizer.backend_tokenizer.normalizer.normalize_str(x)
                for x in lines
            ]
            spec_pre_tokenized_seqs = [
                spec_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(x)[0][0]
                for x in spec_normalized_seqs
            ]
            logging.info("Spec normalizing/tokenizing sequences finished.")
            self.specs = [
                [
                    re.findall(rf"(?<={specifiersep})[^{specifiersep}]+", y)
                    for y in x.split(" ")
                ]
                for x in spec_pre_tokenized_seqs
            ]
            self.nspecs = len(max(max([x for x in y]) for y in self.specs))
            self.specs = [
                np.array(
                    [
                        np.pad(
                            list(map(float, y)),
                            (0, self.nspecs - len(y)),
                            constant_values=0.0,
                        )
                        for y in x[: tokenizer.model_max_length]
                    ]
                )
                for x in self.specs
            ]
            self.specs = [
                np.pad(
                    x,
                    ((0, tokenizer.model_max_length - x.shape[0]), (0, 0)),
                    constant_values=0,
                )
                for x in self.specs
            ]

        pre_tokenized_seqs = [
            tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(x)
            for x in normalized_seqs
        ]
        logging.info("Pre-tokenizing sequences finished.")
        self.seqs = [
            self.join_str.join([y[0] for y in x]).replace("Ġ", "")
            for x in pre_tokenized_seqs
        ]

        # Set the log level to error to supress the warning that we will
        # actually tokenize sequences which are longer than the model's max sequence length.
        log_lvl = transformers.utils.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        # Evaluate the length of the tokenized unmanipulated/untruncated data.
        if self.encoding in ["3mer", "5mer"]:
            self.seqs = self.tokenize_kmers(self.seqs, args)
            raw_encodings = self.tokenizer(
                self.seqs,
                add_special_tokens=False,
                truncation=False,
                is_split_into_words=True,
            )["input_ids"]
        else:
            raw_encodings = self.tokenizer(
                self.seqs, add_special_tokens=False, truncation=False
            )["input_ids"]
        logging.info("Raw tokenizing sequences finished.")
        # restore log lvl
        transformers.logging.set_verbosity(log_lvl)
        self.tokenized_seqs = [
            self.tokenizer.convert_ids_to_tokens(x) for x in raw_encodings
        ]
        logging.info("Re-builiding tokenized sequences finished.")
        self.tokenized_seqs = [
            list(map(lambda x: x.replace("Ġ", ""), y)) for y in self.tokenized_seqs
        ]
        self.tokenized_seqs = [[x for x in y if x != ""] for y in self.tokenized_seqs]

        # Possible cds-centering.
        if self.centertoken:
            # XXX Check this if that still also holds for k-mers.
            self.seqs = self.get_centered_lines()

        # These two options are actively filtering sequences out and also alter `self.lines`.
        if self.only512:
            self.seqs = self.get_only512()
        if self._3utr:
            self.seqs = self.get_3utr()

        if self.non3utr:
            self.seqs = self.get_non3utr()

        if self.nomarkers:
            self.seqs = self.get_nomarkers()

        encodings = self.tokenizer(
            self.seqs,
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding="max_length",
            is_split_into_words=self.encoding in ["3mer", "5mer"],
        )["input_ids"]
        logging.info("Encoding sequences finished.")

        self.examples = np.array([{"input_ids": e} for e in encodings])

        # TODO: Make this a model attribute
        # Set up the scaler
        scaling = getattr(
            getattr(args, "training", None), "scaling", getattr(args, "scaling", None)
        )
        if scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "log":
            self.scaler = LogScaler()
        else:
            # Not so pretty, but is currently the fastest adaptation for no scaling
            self.scaler = IdentityScaler()

        # get the labels and seq idx for each task.
        mode = getattr(args, "mode", None)
        labelpos = ds_get("labelpos", getattr(args, "labelpos", None))
        weightpos = getattr(args, "weightpos", None)
        if mode in ["fine-tune", "predict", "interpret"] and labelpos is not None:
            if args.task == "regression":
                labels = [
                    float(x.split(columnsep)[labelpos - 1].strip('"'))
                    for x in self.lines
                ]
                if weightpos is not None:
                    qualities = [x.split(",")[weightpos].strip('"') for x in self.lines]
                    qual_dict = {"STRONG": 1.0, "GOOD": 0.75, "WEAK": 0.5, "POOR": 0.25}
                    self.qualities = [qual_dict[x] for x in qualities]

                self.labels = self.scaler.fit_transform(
                    np.array(labels).reshape(-1, 1).astype(float)
                )
            elif getattr(args, "task", None) == "classification":
                labels = [
                    x.split(columnsep)[labelpos - 1].strip('"') for x in self.lines
                ]
                self.labels = self.LE.fit_transform(labels)

            # update self.examples with labels (and quality weights).
            if weightpos is None:
                for l, e in zip(self.labels, self.examples):
                    e.update({"labels": l})
            elif getattr(args, "data", None) == "protein":
                for l, e, q in zip(self.labels, self.examples, self.qualities):
                    e.update({"labels": l})
                    e.update({"qualities": q})

    def __len__(self):
        return len(self.examples)

    def get_centered_lines(self):
        centered_lines = list()
        for line in self.tokenized_seqs:
            if len(line) > self.tokenizer.model_max_length:
                cds_pos = [i for i, x in enumerate(line) if self.centertoken in x]
                if cds_pos:
                    cds_pos = cds_pos[0]
                    middle = (self.tokenizer.model_max_length - 2) // 2
                    if cds_pos >= middle:
                        rest_right = max(0, middle - (len(line) - cds_pos))
                        line = line[max(0, cds_pos - middle - rest_right) :]
            centered_lines.append(line)
            if self.encoding not in ["3mer", "5mer"]:
                centered_lines[-1] = self.join_str.join(centered_lines[-1])
        return centered_lines

    def get_only512(self):
        lines = list()
        raw_lines = list()
        for line, raw_line in zip(self.tokenized_seqs, self.lines):
            if len(line) <= self.tokenizer.model_max_length:
                lines.append("".join(line))
                raw_lines.append(raw_line)
        self.lines = raw_lines
        return lines

    def get_3utr(self):
        lines = list()
        raw_lines = list()
        for line, raw_line in zip(self.normalized_lines, self.lines):
            cds_pos = [i for i, x in enumerate(line) if x == "e"]
            if not cds_pos:
                continue
            cds_pos = cds_pos[0]
            line = line[cds_pos + 1 :]
            lines.append(line)
            raw_lines.append(raw_line)
        self.lines = raw_lines
        return lines

    def get_non3utr(self):
        _lines = list()
        for line in self.normalized_lines:
            cds_pos = [i for i, x in enumerate(line) if x == "e"]
            if not cds_pos:
                _lines.append(line)
                continue
            cds_pos = cds_pos[0]
            line = line[:cds_pos]
            _lines.append(line)
        return _lines

    def get_nomarkers(self):
        _lines = list()
        for line in self.normalized_lines:
            line = re.sub("s|e|x", "", line)
            _lines.append(line)
        return _lines

    def log_raw_data(self):
        raw_data_df = pd.DataFrame()
        raw_data_df["seq"] = self.tokenized_seqs
        raw_data_df["lengths"] = raw_data_df["seq"].apply(lambda x: len(x))

        logging.info("Dataset raw statistics:")
        logging.info(raw_data_df.describe(include="all"))

    def log_data(self):
        data_df = pd.DataFrame()
        data_df["seq"] = [
            self.tokenizer.convert_ids_to_tokens(x["input_ids"]) for x in self.examples
        ]
        data_df["lengths"] = data_df["seq"].apply(lambda x: len(x))
        if getattr(self.args, "mode", None) in ["fine-tune", "predict", "interpret"]:
            data_df["labels"] = self.labels
        logging.info("Dataset statistics after truncation and adding special tokens:")
        logging.info(data_df.describe(include="all"))

    @staticmethod
    def tokenize_kmers(lines, args):
        """
        This method is also called when training tokenizers with `learn_tokenizer.py`,
        so we make it static.
        """
        split_lines = list()
        # Support both the structured BioLMConfig and legacy flat top-level attributes
        tokenization = getattr(args, "tokenization", None)
        data_source = getattr(args, "data_source", None)
        settings = getattr(args, "settings", None)

        def tk_get(key, default=None):
            if tokenization is not None and hasattr(tokenization, key):
                return getattr(tokenization, key)
            return getattr(args, key, default)

        def ds_get(key, default=None):
            if data_source is not None and hasattr(data_source, key):
                return getattr(data_source, key)
            return getattr(args, key, default)

        def settings_get(key, default=None):
            if settings is not None:
                dp = getattr(settings, "data_pre_processing", None)
                if isinstance(dp, dict) and key in dp:
                    return dp.get(key)
            return getattr(args, key, default)

        if tk_get("encoding", getattr(args, "encoding", None)) == "3mer":
            pattern = "s|[^xs]{3}|[^xs]{2}x[^xs]|[^xs]x[^xs]{2}|x"
        else:
            pattern = "s|[^xs]{5}|[^xs]{4}x[^xs]|[^xs]x[^xs]{4}||[^xs]{2}x[^xs]{3}|[^xs]{3}x[^xs]{2}|x"
        for line in lines:
            from biolm_utils.tokenization_helpers import parse_atomic_replacements

            atomicreplacements = tk_get(
                "atomicreplacements", getattr(args, "atomicreplacements", None)
            )
            rep = parse_atomic_replacements(atomicreplacements)
            if rep is not None:
                for k, v in rep.items():
                    tokensep = ds_get("tokensep", getattr(args, "tokensep", None))
                    if tokensep is not None:
                        line = line.replace(
                            f"{tokensep}{k}{tokensep}",
                            f"{tokensep}{v}{tokensep}",
                        )
                        line = line.replace(f"\n{k}{tokensep}", f"\n{v}{tokensep}")
                        line = line.replace(f"{tokensep}{k}\n", f"{tokensep}{v}\n")
                    else:
                        line = line.replace(k, v)
            centertoken = settings_get(
                "centertoken", getattr(args, "centertoken", None)
            )
            cds_end_pos = [i for i, x in enumerate(line) if x == centertoken]
            if not cds_end_pos:
                split_lines.append(re.findall(pattern, line))
                continue
            else:
                cds_end_pos = cds_end_pos[0]
                front = line[:cds_end_pos]
                back = line[cds_end_pos + 1 :]
                split_front = re.findall(pattern, front[::-1])[::-1]
                split_front = [x[::-1] for x in split_front]
                split_back = re.findall(pattern, back)
                split_line = split_front + ["s"] + split_back
                split_lines.append(split_line)
        return split_lines

    def __getitem__(example):
        raise NotImplementedError
