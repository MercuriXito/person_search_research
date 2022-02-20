import torch
import os

from utils.misc import pickle
from evaluation.evaluator import FastGraphPSEvaluator, GraphPSEvaluator
from evaluation.eval import FasterRCNNExtractor, evaluate, \
    GTFeatureExtractor
from evaluation.eval_defaults import build_and_load_from_dir \
    as build_and_load_from_dir


def main():
    """ Evaluation with GraphNet which includes ACAE branch.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--eval-config", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    model, t_args = build_and_load_from_dir(args.exp_dir, args.eval_config, args.opts)
    eval_args = t_args.eval
    checkpoint_path = os.path.join(args.exp_dir, eval_args.checkpoint)

    device = torch.device(eval_args.device)
    if eval_args.use_gt:
        extractor = GTFeatureExtractor(model, device)
    else:
        extractor = FasterRCNNExtractor(model, device)

    if eval_args.use_fast_graph:
        ps_evaluator = FastGraphPSEvaluator(
            model.graph_head, device, eval_args.dataset_file,
            topk=eval_args.fast_graph_topk,
            eval_all_sim=eval_args.eval_all_sim)
    else:
        ps_evaluator = GraphPSEvaluator(
            model.graph_head, device, eval_args.dataset_file,
            eval_all_sim=eval_args.eval_all_sim)

    res_pkl, table_string = evaluate(
        extractor, eval_args, ps_evaluator=ps_evaluator)

    # serealization
    prefix = "eval"
    if eval_args.eval_context:
        prefix = f"acae.G{eval_args.graph_thred}.{prefix}"
    if eval_args.use_gt:
        prefix = f"{prefix}.gt"
    save_path = f"{checkpoint_path}.{prefix}.pkl"
    table_string_path = f"{checkpoint_path}.{prefix}.txt"
    pickle(res_pkl, save_path)
    with open(table_string_path, "w") as f:
        f.write(table_string)


if __name__ == '__main__':
    main()
