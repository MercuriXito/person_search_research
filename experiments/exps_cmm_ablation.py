""" Ablation experiments of CMM:
Effects of each components.
"""
import os
import torch
import argparse
import numpy as np
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment

from utils.misc import unpickle
from evaluation.eval_defaults import build_and_load_from_dir
from evaluation.evaluator import PersonSearchEvaluator, get_cosine_sim
from evaluation.eval import FasterRCNNExtractor, evaluate


def format_eval_res(res_data):
    formats = ["{}"] + ["{:8f}"] * 8
    format_eval_res = [
        formats[i].format(res_item)
        for i, res_item in enumerate(res_data)]
    return format_eval_res


def graph_dist(image_match_scores, reid_scores, do_filter=True):
    """ adaptive graph distance, excluding the context from gallery target.
    """
    if not do_filter:
        if image_match_scores.size == 0:
            return reid_scores
        return_scores = [image_match_scores.mean() for _ in range(len(reid_scores))]
        scores = np.array(return_scores)[:, None]
        if np.isnan(scores).any():
            from IPython import embed
            embed()
        return scores

    return_scores = []
    for idx in range(len(reid_scores)):
        reid_score = reid_scores[idx, 0]
        valid_pos_mask = np.ones_like(image_match_scores)
        valid_pos_mask[idx, :] = 0
        valid_pos_mask = valid_pos_mask.astype(np.bool)
        valid_context = image_match_scores > reid_score
        valid_context = np.logical_and(valid_context, valid_pos_mask)
        if valid_context.sum() == 0:
            return_scores.append(float(reid_score))
        else:
            return_scores.append(float(image_match_scores[valid_context].mean()))
    return np.array(return_scores)[:, None]


def get_ablation_context_sim(
                    gallery_feat,
                    query_feat,
                    graph_thred,
                    do_filter=True,
                    do_nmss=True,
                    *args, **kwargs):
    """ Context Similarity between features in query images and gallery images.
    used in ablation study.
    """
    # HACK: the last ones in query features must be the target features.
    idx = -1

    query_context_feat = query_feat[:idx, :]
    query_target_feat = query_feat[idx, :][None]

    indv_scores = np.matmul(gallery_feat, query_target_feat.T)
    if len(query_target_feat) == 0:
        return indv_scores
    sim_matrix = np.matmul(gallery_feat, query_context_feat.T)

    # contextual scores
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    qg_mask = np.zeros_like(sim_matrix)
    qg_mask[row_ind, col_ind] = 1
    qg_sim_matrix = sim_matrix * qg_mask
    graph_scores = graph_dist(qg_sim_matrix, indv_scores, do_filter=do_filter)
    final_scores = indv_scores * (1 - graph_thred) + graph_scores * graph_thred

    if np.isnan(final_scores).any():
        from IPython import embed
        embed()

    # split scores
    if do_nmss:
        final_scores = torch.as_tensor(final_scores)
        final_scores_softmax = torch.softmax(final_scores, 0)
        final_scores = final_scores_softmax * final_scores / final_scores_softmax.max()
        final_scores = np.array(final_scores.cpu())
    return final_scores


class CMMAblationEvaluator(PersonSearchEvaluator):
    def __init__(self, dataset_file="cuhk-sysu", options=None) -> None:
        super().__init__(dataset_file)
        if options is None:
            self.options = {}
        else:
            assert isinstance(options, dict)
            self.options = options

    def get_similarity(
            self, gallery_feat, query_feat, use_context=True, graph_thred=0.0):
        if len(query_feat.shape) == 1:
            query_feat = query_feat.reshape(1, -1)
        if len(gallery_feat.shape) == 1:
            gallery_feat = gallery_feat.reshape(1, -1)

        if not use_context:
            query_target_feat = query_feat[-1].reshape(1, -1)
            return get_cosine_sim(gallery_feat, query_target_feat)
        return get_ablation_context_sim(
            gallery_feat, query_feat, graph_thred=graph_thred,
            **self.options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--eval-config", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = os.sep.join(args.pickle.split(os.sep)[:-1])
    model, t_args = build_and_load_from_dir(exp_dir, args.eval_config, args.opts)
    assert t_args.eval.dataset_file == "cuhk-sysu", "gallery size opt is only available in CUHK-SYSU."
    eval_args = t_args.eval
    device = torch.device(eval_args.device)
    extractor = FasterRCNNExtractor(model, device)
    checkpoint_path = os.path.join(exp_dir, eval_args.checkpoint)

    # infos in pickle
    data = unpickle(args.pickle)
    eval_res = data["eval_res"]
    file_names = list(eval_res.keys())
    file_names[0] = "CMM options"
    file_res = list(eval_res.values())
    file_res[0] = "res in pickle"
    table = PrettyTable(field_names=file_names)
    table.add_row(format_eval_res(file_res))
    print("Eval results in pickle({}) :".format(args.pickle))
    print(table)

    # evaluate with different ablation options:
    ablation_options = [
        [False, False],
        [True, False],
        [True, True]
    ]

    for options in ablation_options:
        eval_options = {
            "do_filter": options[0],
            "do_nmss": options[1]
        }

        # choose evaluator
        evaluator = CMMAblationEvaluator(eval_args.dataset_file, options=eval_options)

        res_pkl, _ = evaluate(
            extractor, eval_args, ps_evaluator=evaluator, res_pkl=data)
        eval_res = list(res_pkl["eval_res"].values())
        eval_res[0] = ";".join([f"{k}={v}" for k, v in eval_options.items()])
        table.add_row(format_eval_res(eval_res))
        print(table)

        table_string_path = f"{checkpoint_path}.cmm_ablation.txt"
        with open(table_string_path, "w") as f:
            f.write(table.get_string())


if __name__ == '__main__':
    main()
