""" Experiments fast graph:
explore the effect of k in fast graph
"""
import os
import argparse
import torch
from prettytable import PrettyTable

from utils.misc import unpickle
from evaluation.eval import FasterRCNNExtractor, evaluate
from evaluation.evaluator import FastGraphPSEvaluator
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
    assert model.graph_head is not None, "Exps only for ACAE."
    device = torch.device(eval_args.device)
    extractor = FasterRCNNExtractor(model, device)
    checkpoint_path = os.path.join(exp_dir, eval_args.checkpoint)

    # infos in pickle
    data = unpickle(args.pickle)
    eval_res = data["eval_res"]
    file_names = list(eval_res.keys())
    file_names[0] = "gallery_size"
    file_res = list(eval_res.values())
    file_res[0] = "res in pickle"
    table = PrettyTable(field_names=file_names)
    table.add_row(format_eval_res(file_res))
    print("Eval results in pickle({}) :".format(args.pickle))
    print(table)

    # evaluate with different gallery_size:
    fast_graph_ks = [50, 100, 200]
    for fast_graph_k in fast_graph_ks:

        # modify lambda
        eval_args.defrost()
        eval_args.use_fast_graph = True
        eval_args.fast_graph_topk = fast_graph_k
        eval_args.freeze()

        evaluator = FastGraphPSEvaluator(model.graph_head, device,
                                         eval_args.dataset_file,
                                         topk=fast_graph_k,
                                         eval_all_sim=eval_args.eval_all_sim)

        res_pkl, _ = evaluate(
            extractor, eval_args, ps_evaluator=evaluator, res_pkl=data)
        eval_res = list(res_pkl["eval_res"].values())
        eval_res[0] = f"{fast_graph_k:2.2f}"
        table.add_row(format_eval_res(eval_res))
        print(table)

        table_string_path = f"{checkpoint_path}.fast_graphk.txt"
        with open(table_string_path, "w") as f:
            f.write(table.get_string())


if __name__ == '__main__':
    main()
