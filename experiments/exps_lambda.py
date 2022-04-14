""" Experiments lambda:
explore the effect of different lambda.
"""
import os
import argparse
import torch
import numpy as np
from prettytable import PrettyTable

from utils.misc import unpickle
from evaluation.eval import FasterRCNNExtractor, evaluate
from evaluation.evaluator import PersonSearchEvaluator, GraphPSEvaluator
from evaluation.eval_defaults import build_and_load_from_dir


def format_eval_res(res_data):
    formats = ["{}"] + ["{:8f}"] * 8
    format_eval_res = [
        formats[i].format(res_item)
        for i, res_item in enumerate(res_data)]
    return format_eval_res


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
    file_res = list(eval_res.values())
    file_res[0] = "res in pickle"
    table = PrettyTable(field_names=file_names)
    table.add_row(format_eval_res(file_res))
    print("Eval results in pickle({}) :".format(args.pickle))
    print(table)

    # evaluate with different lambda:
    step = 0.1
    graph_threshs = np.arange(0, 1.0 + step, step).tolist()
    for graph_thresh in graph_threshs:

        # modify lambda
        eval_args.defrost()
        eval_args.graph_thred = graph_thresh
        eval_args.freeze()

        # choose evaluator
        if t_args.model.graph_head.use_graph:
            evaluator = GraphPSEvaluator(model.graph_head, device,
                                         eval_args.dataset_file,
                                         eval_all_sim=eval_args.eval_all_sim)
        else:
            evaluator = PersonSearchEvaluator(eval_args.dataset_file)

        res_pkl, _ = evaluate(
            extractor, eval_args, ps_evaluator=evaluator, res_pkl=data)
        eval_res = list(res_pkl["eval_res"].values())
        eval_res[0] = f"G={graph_thresh:2.2f}"
        table.add_row(format_eval_res(eval_res))
        print(table)

        table_string_path = f"{checkpoint_path}.graph_thresh.txt"
        with open(table_string_path, "w") as f:
            f.write(table.get_string())


if __name__ == '__main__':
    main()
