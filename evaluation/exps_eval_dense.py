import os
import torch
from easydict import EasyDict
from prettytable import PrettyTable
from copy import deepcopy

from datasets import load_eval_datasets
from evaluation.eval import FasterRCNNExtractor, evaluate
from evaluation.evaluator import GraphPSEvaluator, PersonSearchEvaluator
from evaluation.eval_defaults import build_and_load_from_dir
from evaluation.badcase import get_all_query_persons


def get_dense_dataset(select_thresh=10):
    """ only support cuhk-sysu dataset right now.
    """
    dataset = load_eval_datasets(EasyDict(
        dataset_file="cuhk-sysu",
        root="data/cuhk-sysu/"
    ))
    probes = dataset.probes
    query_boxes = get_all_query_persons(dataset)

    # select probes under dense situation.
    dense_probes = []
    for i, boxes in enumerate(query_boxes):
        probes[i]["search_idx"] = i
        if len(boxes) >= select_thresh:
            dense_probes.append(probes[i])

    setattr(dataset, "probes", dense_probes)
    setattr(dataset, "original_probes", probes)
    return dataset


def evaluate_dense():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_exp_dir")
    parser.add_argument("graph_exp_dir")
    parser.add_argument("--dataset_file", default="cuhk-sysu", choices=["cuhk-sysu", "prw"])
    args = parser.parse_args()

    baseline, b_args = build_and_load_from_dir(args.baseline_exp_dir)
    graph, g_args = build_and_load_from_dir(args.graph_exp_dir)

    device = torch.device("cuda:0")
    b_extractor = FasterRCNNExtractor(baseline, device)
    g_extractor = FasterRCNNExtractor(graph, device)
    b_evaluator = PersonSearchEvaluator(args.dataset_file)
    g_evaluator = GraphPSEvaluator(
        graph.graph_head, device, args.dataset_file,
        eval_all_sim=g_args.eval.eval_all_sim)

    table = PrettyTable(field_names=[
        "item", "model", "det_ap", "det_recall",
        "labeled_ap", "labeled_recall",
        "mAP", "top1", "top5", "top10"])

    dense_threshs = [2, 4, 6, 10, 15] + [18, 20, 25, 30, 45, 50]
    num_query_thresh = 40
    for thresh in dense_threshs:
        imdb = get_dense_dataset(thresh)
        print("Thresh {}: {}".format(thresh, len(imdb.probes)))

        if len(imdb.probes) < num_query_thresh:
            print("Thresh {} not enough querys".format(thresh))
            continue

        b_args.defrost()

        # performance of baseline
        # 1. with CMM.
        # eval_args = deepcopy(b_args.eval)

        # eval_args.eval_context = True
        # eval_args.graph_thresh = 0.4
        # res, _ = evaluate(b_extractor, eval_args, imdb=imdb, ps_evaluator=b_evaluator)
        # baseline_res = res["eval_res"]
        # format_eval_res = ["{:.8f}".format(res_item) for res_item in baseline_res]
        # format_eval_res = [f"{thresh:.2f}", f"baseline:CMM{eval_args.graph_thresh}"] + format_eval_res
        # table.add_row(format_eval_res)

        # # 2. w/o CMM
        eval_args.eval_context = False
        eval_args.graph_thresh = 0.0
        res, _ = evaluate(
            b_extractor, eval_args, imdb=imdb, ps_evaluator=b_evaluator,
            res_pkl=res)
        baseline_res = res["eval_res"]
        format_eval_res = ["{:.8f}".format(res_item) for res_item in baseline_res]
        format_eval_res = [f"{thresh:.2f}", "baseline"] + format_eval_res
        table.add_row(format_eval_res)

        # To fairly compare the performance between ACAE
        # and CMM performance
        # performance of graph counterpart
        # 1. with ACAE
        g_args.defrost()
        eval_args = deepcopy(g_args.eval)

        eval_args.eval_context = True
        eval_args.graph_thresh = 0.4
        res, _ = evaluate(
            g_extractor, eval_args, imdb=imdb, ps_evaluator=g_evaluator,
            res_pkl=res)  # evaluation with the same previous features
        graph_res = res["eval_res"]
        format_eval_res = ["{:.8f}".format(res_item) for res_item in graph_res]
        format_eval_res = [f"{thresh:.2f}", "Graph(0.4)"] + format_eval_res
        table.add_row(format_eval_res)

        # # 2. w/o ACAE
        # eval_args.eval_context = False
        # eval_args.graph_thresh = 0.0
        # res, _ = evaluate(
        #     g_extractor, eval_args, imdb=imdb,
        #     ps_evaluator=g_evaluator, res_pkl=res)
        # graph_res = res["eval_res"]
        # format_eval_res = ["{:.8f}".format(res_item) for res_item in graph_res]
        # format_eval_res = [f"{thresh:.2f}", "Graph(0.0)"] + format_eval_res
        # table.add_row(format_eval_res)

        # # 3. graph baseline but with CMM(0.4)
        # eval_args.eval_context = True
        # eval_args.graph_thresh = 0.4
        # res, _ = evaluate(
        #     g_extractor, eval_args, imdb=imdb,
        #     ps_evaluator=b_evaluator, res_pkl=res)
        # graph_res = res["eval_res"]
        # format_eval_res = ["{:.8f}".format(res_item) for res_item in graph_res]
        # format_eval_res = [f"{thresh:.2f}", "Graph(0.4)"] + format_eval_res
        # table.add_row(format_eval_res)

        print(table)

    save_file = os.path.join("exps", "graph_baseline.dense.res.txt")
    with open(save_file, "w") as f:
        f.write(table.get_string())


if __name__ == '__main__':
    evaluate_dense()
