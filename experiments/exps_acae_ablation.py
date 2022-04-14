""" Ablation Experiments of ACAE:
- Effects of each components
- Effects of different features
"""
import os
import torch
import argparse
import numpy as np
from prettytable import PrettyTable

from utils.misc import unpickle
from evaluation.eval_defaults import build_and_load_from_dir
from evaluation.evaluator import GraphPSEvaluator, get_cosine_sim
from evaluation.eval import FasterRCNNExtractor, evaluate


def format_eval_res(res_data):
    formats = ["{}"] + ["{:8f}"] * 8
    format_eval_res = [
        formats[i].format(res_item)
        for i, res_item in enumerate(res_data)]
    return format_eval_res


class AblationGraphPSEvaluator(GraphPSEvaluator):
    def __init__(self, graph_head, device, dataset_file="cuhk-sysu",
                 sim_weights: list = [1, 1, 1],
                 **eval_kwargs) -> None:
        super().__init__(graph_head, device, dataset_file, **eval_kwargs)
        assert len(sim_weights) == 3
        for w in sim_weights:
            assert w >= 0.0 and w <= 1.0, f"{w} should be in range [0, 1]"
        self.sim_weights = torch.tensor(sim_weights).to(self.device)

    def get_similarity(
            self, gallery_feat, query_feat, use_context, graph_thred,
            **eval_kwargs):
        if len(query_feat.shape) == 1:
            query_feat = query_feat.reshape(1, -1)
        if len(gallery_feat.shape) == 1:
            gallery_feat = gallery_feat.reshape(1, -1)

        if not use_context:
            query_target_feat = query_feat[-1].reshape(1, -1)
            return get_cosine_sim(gallery_feat, query_target_feat)

        idx = -1
        query_context_feat = query_feat[:idx, :]
        query_target_feat = query_feat[idx, :][None]

        indv_scores = np.matmul(gallery_feat, query_target_feat.T)
        if len(query_context_feat) == 0:
            return indv_scores

        with torch.no_grad():
            gallery_feat = torch.as_tensor(gallery_feat).to(self.device)
            query_context_feat = torch.as_tensor(query_context_feat).to(self.device)
            query_target_feat = torch.as_tensor(query_target_feat).to(self.device)
            scores = self.graph_head.inference(
                gallery_feat, query_context_feat, query_target_feat,
                graph_thred=graph_thred,
                eval_all_sim=self.eval_all_sim
            )

            # graph head inference
            reid_scores = torch.matmul(gallery_feat, query_target_feat.T)
            if len(query_context_feat) == 0:
                return reid_scores

            gfeats = gallery_feat
            qfeats = torch.cat([query_context_feat, query_target_feat])
            # TODO: check if the wrong order does harm to effects.
            all_gfeats, all_qfeats = self.graph_head.graph_head.inference_features(gfeats, qfeats)
            sim_matrix = self.get_context_sim(all_gfeats, all_qfeats)

            # NMSS
            graph_scores = sim_matrix[:, -1][..., None]
            scores = graph_thred * graph_scores + (1 - graph_thred) * reid_scores
            sscores = torch.softmax(scores, dim=0)
            scores = sscores * scores / sscores.max()

        scores = scores.cpu().numpy()
        return scores

    def get_context_sim(self, all_gfeats, all_qfeats):
        assert len(all_gfeats) == 3
        assert len(all_qfeats) == 3

        gfeats, gself_feats, gcross_feats = all_gfeats
        qfeats, qself_feats, qcross_feats = all_qfeats

        all_sim_mat = [
            torch.matmul(gfeats, qfeats.T),
            torch.matmul(gself_feats, qself_feats.T),
            torch.matmul(gcross_feats, qcross_feats.T)
        ]
        all_sim_mat = torch.stack(all_sim_mat, dim=0)
        all_sim_mat = (all_sim_mat * self.sim_weights.view(3, 1, 1)).mean(dim=0)
        return all_sim_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--eval-config", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = os.sep.join(args.pickle.split(os.sep)[:-1])
    model, t_args = build_and_load_from_dir(exp_dir, args.eval_config, args.opts)
    eval_args = t_args.eval
    device = torch.device(eval_args.device)
    extractor = FasterRCNNExtractor(model, device)
    checkpoint_path = os.path.join(exp_dir, eval_args.checkpoint)

    # infos in pickle
    data = unpickle(args.pickle)
    eval_res = data["eval_res"]
    file_names = list(eval_res.keys())
    file_names[0] = "ACAE options"
    file_res = list(eval_res.values())
    file_res[0] = "res in pickle"
    table = PrettyTable(field_names=file_names)
    table.add_row(format_eval_res(file_res))
    print("Eval results in pickle({}) :".format(args.pickle))
    print(table)

    # evaluate with different ablation options:
    names = ["final", "self", "cross"]
    weights_options = [
        [1, 1, 1],  # all
        [0, 1, 1],  # final-except
        [1, 0, 1],  # cross-except
        [1, 1, 0],  # self-except
        [0, 0, 1],  # self-only
        [0, 1, 0],  # cross-only
        [1, 0, 0],  # final-only
    ]

    for sim_weights in weights_options:
        # choose evaluator
        evaluator = AblationGraphPSEvaluator(
            model.graph_head, device, eval_args.dataset_file,
            eval_all_sim=eval_args.eval_all_sim, sim_weights=sim_weights)

        res_pkl, _ = evaluate(
            extractor, eval_args, ps_evaluator=evaluator, res_pkl=data)
        eval_res = list(res_pkl["eval_res"].values())
        eval_res[0] = ";".join([f"{k}={v}" for k, v in zip(names, sim_weights) ])
        table.add_row(format_eval_res(eval_res))
        print(table)

        table_string_path = f"{checkpoint_path}.acae_features_ablation.txt"
        with open(table_string_path, "w") as f:
            f.write(table.get_string())


if __name__ == '__main__':
    main()
