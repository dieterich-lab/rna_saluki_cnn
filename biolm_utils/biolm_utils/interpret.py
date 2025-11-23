import logging
import pickle
import warnings

import pandas as pd
import torch

# see: https://github.com/shap/shap/issues/2909
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import shap

from biolm_utils.config import get_config
from biolm_utils.loo_utils import TauLOO_Evaluation_For_Regression
from biolm_utils.train_utils import get_model_and_config


def loo_scores(
    args,
    tokenizer,
    model_cls,
    test_dataset,
    model_load_path,
    output_path,
    remove_first_last,
):
    config = get_config()

    # Structured-config compatibility helpers
    data_source = getattr(args, "data_source", None)
    tokenization = getattr(args, "tokenization", None)
    inference_cfg = getattr(args, "inference", None)
    debugging = getattr(args, "debugging", None)

    def ds_get(key, default=None):
        if data_source is not None and hasattr(data_source, key):
            return getattr(data_source, key)
        return getattr(args, key, default)

    def tk_get(key, default=None):
        if tokenization is not None and hasattr(tokenization, key):
            return getattr(tokenization, key)
        return getattr(args, key, default)

    def debug_get(key, default=None):
        if debugging is not None and hasattr(debugging, key):
            return getattr(debugging, key)
        return getattr(args, key, default)

    def inf_loose_get(key, default=None):
        if inference_cfg is not None:
            loos = getattr(inference_cfg, "looscores", None)
            if isinstance(loos, dict) and key in loos:
                return loos.get(key)
        return getattr(args, key, default)

    pkl_file = (
        output_path
        / f"loo_scores_{inf_loose_get('handletokens', getattr(args, 'handletokens', None))}.pkl"
    )
    csv_file = (
        output_path
        / f"loo_scores_{inf_loose_get('handletokens', getattr(args, 'handletokens', None))}.csv"
    )

    test_idx = test_dataset.indices

    nlabels = 1

    # Getting the model and config.
    model = get_model_and_config(
        args=args,
        model_cls=model_cls,
        model_config_cls=config.CONFIG_CLS,
        tokenizer=tokenizer,
        dataset=test_dataset,
        nlabels=nlabels,
        model_load_path=model_load_path,
        pretraining_required=config.PRETRAINING_REQUIRED,
        scaler=None,
    )

    # Send the model to the proper device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loo_scores = list()

    tl = TauLOO_Evaluation_For_Regression(
        model,
        tokenizer,
        OHE=test_dataset.dataset.OHE,
        specs=test_dataset.dataset.specs,
        specifiersep=ds_get("specifiersep", getattr(args, "specifiersep", None)),
        tokensep=ds_get("tokensep", getattr(args, "tokensep", None)),
    )

    for i, test_id in enumerate(
        test_idx[
            : len(test_idx) if not debug_get("dev", False) else debug_get("dev", False)
        ]
    ):
        seq = test_dataset.dataset.lines[test_id]
        loo_score, rescaled_pred, token_list, replacements = (
            tl.compute_leave_one_out_occlusion(
                text=seq,
                id=test_id,
                remove_first_last=remove_first_last,
                handle_tokens=inf_loose_get(
                    "handletokens", getattr(args, "handletokens", None)
                ),
                scaler=model.scaler,
                batch_size=getattr(
                    getattr(args, "training", None),
                    "batchsize",
                    getattr(args, "batchsize", None),
                ),
                replacement_dict=inf_loose_get(
                    "replacementdict", getattr(args, "replacementdict", None)
                ),
                replacespecifier=inf_loose_get(
                    "replacespecifier", getattr(args, "replacespecifier", None)
                ),
                dev=debug_get("dev", False),
            )
        )
        token_list = [x.replace("Ġ", "") for x in token_list]
        scores = list()
        if replacements is not None:
            score_id = 0
            for j, (token, replacement) in enumerate(zip(token_list, replacements)):
                for k, rep in enumerate(replacement):
                    scores.append(
                        (token, rep, loo_score[score_id].item(), rescaled_pred, j)
                    )
                    score_id += 1
            loo_scores.append(scores)
        else:
            # Here, we'll save the scores conveniently as `shap.Explanation` object.
            exp = shap.Explanation(
                loo_score[None, :], rescaled_pred, feature_names=token_list
            )
            loo_scores.append(exp)

        label = test_dataset.dataset[test_id]["labels"]
        rescaled_label = model.scaler.inverse_transform(label)
        seq = test_dataset.dataset.seq_idx[test_id]
        if replacements is not None:
            token_list = [x[0] for x in loo_scores[-1]]
            reps = [x[1] for x in loo_scores[-1]]
            loos = [x[2] for x in loo_scores[-1]]
            preds = [x[3].item() for x in loo_scores[-1]]
            poss = [x[4] for x in loo_scores[-1]]
            seqs = [seq] * len(loos)
            labels = [rescaled_label.item()] * len(loos)
        else:
            loos = loo_scores[-1].values.tolist()[0]
            seqs = [seq] * loo_scores[-1].shape[1]
            labels = [rescaled_label.item()] * loo_scores[-1].shape[1]
            token_list = loo_scores[-1].feature_names
            preds = list(loo_scores[-1].base_values) * loo_scores[-1].shape[1]
            poss = [0] * len(token_list)  # dummy list
        starts, ends = list(), list()
        pos = 0
        for k, (token, l) in enumerate(zip(token_list, poss)):
            if replacements is not None:
                if tk_get("encoding", getattr(args, "encoding", None)) == "atomic":
                    starts.append(l)
                    ends.append(l + 1)
                else:
                    if k == 0:
                        starts.append(0)
                        ends.append(len(token))
                    else:
                        if poss[k] == pos:
                            starts.append(starts[-1])
                            ends.append(ends[-1])
                        else:
                            pos = poss[k]
                            starts.append(ends[-1])
                            ends.append(ends[-1] + len(token))
            else:
                if k == 0:
                    starts.append(0)
                    ends.append(len(token))
                else:
                    starts.append(ends[-1])
                    ends.append(ends[-1] + len(token))

        if replacements is not None:
            data = list(
                zip(
                    seqs,
                    token_list,
                    reps,
                    labels,
                    preds,
                    starts,
                    ends,
                    loos,
                )
            )
            columns = [
                "sequence",
                "token",
                "replacement",
                "label",
                "pred",
                "start_offset",
                "end_offset",
                "loo",
            ]
        else:
            data = list(
                zip(
                    seqs,
                    token_list,
                    labels,
                    preds,
                    starts,
                    ends,
                    loos,
                )
            )
            columns = [
                "sequence",
                "token",
                "label",
                "pred",
                "start_offset",
                "end_offset",
                "loo",
            ]
        if i == 0:
            logging.info(f"Saving to {csv_file}")
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(csv_file, index=False)
        else:
            logging.info(f"Adding to {csv_file}")
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False, mode="a", header=False)

    logging.info(f"Saving pickled results: {pkl_file}")
    with open(pkl_file, "wb") as f:
        pickle.dump(loo_scores, f)
