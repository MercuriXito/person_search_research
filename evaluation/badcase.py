import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from datasets import load_eval_datasets
from evaluation.eval_defaults import build_and_load_from_dir
from evaluation.evaluator import GraphPSEvaluator, PersonSearchEvaluator
from evaluation.exps_eval_dense import get_all_query_persons
from configs.faster_rcnn_default_configs import get_default_cfg
from utils.misc import unpickle
from utils.vis import compute_ap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="pkl path for evaluation")
    parser.add_argument("--graph", action="store_true")
    return parser.parse_args()


def load_config(exp_dir):
    config_file = os.path.join(exp_dir, "config.yml")
    args = get_default_cfg()
    args.merge_from_file(config_file)
    return args


def load_pickle_and_eval(pkl_path):
    res_pkl = unpickle(pkl_path)
    dirname = os.path.dirname(pkl_path)
    device = torch.device("cuda")
    t_args = load_config(dirname)

    if t_args.model.graph_head.use_graph:
        graph_net, t_args = build_and_load_from_dir(dirname)
        eval_args = t_args.eval
        graph_net.eval()
        graph_net.to(device)
        evaluator = GraphPSEvaluator(graph_net.graph_head, device, eval_args.dataset_file)
    else:
        _, t_args = build_and_load_from_dir(dirname)
        eval_args = t_args.eval
        evaluator = PersonSearchEvaluator(eval_args.dataset_file)
    imdb = load_eval_datasets(eval_args)
    print("Load OK.")

    gallery_boxes = res_pkl["gallery_boxes"]
    gallery_features = res_pkl["gallery_features"]
    query_features = res_pkl["query_features"]

    probes = imdb.probes
    mAP, top1, top5, top10, res_data = evaluator.eval_search(
        imdb, probes, gallery_boxes, gallery_features, query_features,
        det_thresh=eval_args.det_thresh, gallery_size=eval_args.gallery_size,
        use_context=eval_args.eval_context, graph_thred=eval_args.graph_thred)

    return res_data, res_pkl


def is_top1_miss(ret_item):
    """ badcase: gt Top-1 missed.
    """
    return ret_item["gallery"][0]["correct"] != 1


def compare_mAP(ret_item_a, ret_item_b):
    """ badcase: return 1 if mAP(a) is worse than mAP(b)
    meaning that the ranking result of a is inferior than that of b.
    """

    assert ret_item_a["probe_img"] == ret_item_b["probe_img"], \
        "Comparison should be done betweeen the smae probe image"

    rank_list_a = np.array([gitem["correct"] for gitem in ret_item_a["gallery"]])
    rank_list_b = np.array([gitem["correct"] for gitem in ret_item_b["gallery"]])
    # num_gt = len(ret_item_a["probe_gt"])
    num_gt = max(np.sum(rank_list_a), np.sum(rank_list_b)).item()

    apa = compute_ap(rank_list_a, num_gt)
    apb = compute_ap(rank_list_b, num_gt)
    return apa < apb


def main():
    args = get_args()
    pkl_path = args.pkl

    eval_data, pkl_data = load_pickle_and_eval(pkl_path)
    query_persons = get_all_query_persons()

    missed_dets = []
    for idx, item in enumerate(tqdm((eval_data["results"]))):
        if is_top1_miss(item):
            missed_dets.append(len(query_persons[idx]))

    # 统计所有 query 中 person 的数量
    num_querys = defaultdict(lambda: 0)
    for qboxes in query_persons:
        num_querys[len(qboxes)] += 1

    counter = Counter(missed_dets)
    num_missed_ratios = dict()
    for num_appear in num_querys.keys():
        num_missed = counter[num_appear] if num_appear in counter else 0
        num_missed_ratios[num_appear] = num_missed / num_querys[num_appear]

    x = np.array(list(num_missed_ratios.keys()))
    y = np.array(list(num_missed_ratios.values()))

    bars = plt.bar(x, y)
    # plt.xlabel([str(v) for v in x])
    plt.savefig(pkl_path + ".missed_ratio.png")


def main_aps():
    """ query 的 AP 和 query image 中 person 数量的平均趋势
    """
    args = get_args()
    pkl_path = args.pkl

    eval_data, pkl_data = load_pickle_and_eval(pkl_path)
    query_persons = get_all_query_persons()

    data = defaultdict(lambda: [])
    for idx, item in enumerate(tqdm((eval_data["results"]))):
        num_querys = len(query_persons[idx])
        data[num_querys].append(item["probe_ap"])

    x = np.array(list(data.keys()))
    y = np.array([np.asarray(ap_list).mean() for ap_list in data.values()])

    bars = plt.bar(x, y)
    # plt.xlabel([str(v) for v in x])
    plt.title("Averaged ap under scene with different number of persons.")
    plt.xlabel("average ap")
    plt.ylabel("number of persons in query image")
    plt.savefig(pkl_path + ".avg_ap.png")


def main_diff_map():
    """ compare the mAP for each query between two models config
    to find under which circumstance model A is inferior than model B.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl1", required=True, help="pkl path for evaluation")
    parser.add_argument("--pkl2", required=True, help="pkl path for evaluation")
    args = parser.parse_args()

    eval_a, _ = load_pickle_and_eval(args.pkl1)
    eval_b, _ = load_pickle_and_eval(args.pkl2)

    query_persons = get_all_query_persons()
    ans = defaultdict(lambda: [0, 0])

    for idx in tqdm(range(len(eval_a["results"]))):
        qitem_a = eval_a["results"][idx]
        qitem_b = eval_b["results"][idx]

        num_query_person = len(query_persons[idx])
        ans[num_query_person][0] += 1

        if compare_mAP(qitem_a, qitem_b):
            ans[num_query_person][1] += 1

    x = np.array(list(ans.keys()))
    total_persons = np.array([vs[0] for vs in list(ans.values())])
    count_person = np.array([vs[1] for vs in list(ans.values())])

    y = count_person / total_persons

    plt.bar(x, y)
    plt.ylim([0, 1])
    plt.savefig(args.pkl1 + ".bad.png")


if __name__ == '__main__':
    # main()
    # main_diff_map()
    main_aps()
