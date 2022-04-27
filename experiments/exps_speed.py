""" Experiments of inference time:
- define inference time for pair of input
"""
import os
import argparse
import torch
import torch.nn as nn
import time
from datasets import load_eval_datasets
import numpy as np
import random
import PIL.Image as Image
from tqdm import tqdm

from utils.misc import ship_to_cuda, _fix_randomness
from evaluation.eval import FasterRCNNExtractor
from evaluation.evaluator import get_context_sim
from evaluation.eval_defaults import build_and_load_from_dir
from models.graph_net import GraphNet


def time_wrapper(func):

    def _func(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        consume = end_time - start_time
        return consume
    return _func


class SpeedForward(nn.Module):

    def __init__(self, net, graph_head=None, mode="feat"):
        super().__init__()
        self.funcs = {
            "feat": self.feat_forward,
            "baseline": self.baseline_forward,
            "cmm": self.cmm_forward,
            "acae": self.graph_forward,
        }
        self.net = net
        self.graph_head = graph_head
        self.set_mode(mode)

    def set_mode(self, mode):
        assert mode in self.funcs
        self.mode = mode
        self.func_forward = self.funcs[mode]
        self.time_func = time_wrapper(self.func_forward)

    def feat_forward(self, images: list):
        detections, _ = self.net(images)
        return detections

    def baseline_forward(self, images: list):
        detections, _ = self.net(images)
        ind_features = [item["embeddings"] for item in detections]
        sim = torch.matmul(ind_features[0][-1], ind_features[1].T)
        return sim

    def cmm_forward(self, images: list):
        detections, _ = self.net(images)
        ind_features = [item["embeddings"].cpu().numpy() for item in detections]
        sim = get_context_sim(ind_features[1], ind_features[0], graph_thred=0.4)
        return sim

    def graph_forward(self, images: list):
        detections, _ = self.net(images)
        ind_features = [item["embeddings"] for item in detections]
        # ACAE forward
        query_context_feat = ind_features[0][-1, :][None]
        query_target_feat = ind_features[0][:-1, :]
        gallery_feat = ind_features[1]

        scores = self.graph_head.inference(
            gallery_feat, query_context_feat, query_target_feat,
            graph_thred=0.4, eval_all_sim=True
        )
        return scores

    def forward(self, images: list):
        if self.mode == "feat":
            images = [images[0]]
        else:
            assert len(images) % 2 == 0, f"{len(images)} should be even number."

        with torch.no_grad():
            inf_time = self.time_func(images)
        return inf_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--mode", type=str, default="feat",
                        choices=["cmm", "baseline", "acae", "feat"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-config", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = os.sep.join(args.pickle.split(os.sep)[:-1])
    model, t_args = build_and_load_from_dir(exp_dir, args.eval_config, args.opts)
    eval_args = t_args.eval
    assert model.graph_head is not None, "Exps only for ACAE."
    device = torch.device(eval_args.device)
    extractor = FasterRCNNExtractor(model, device)
    imdb = load_eval_datasets(eval_args)

    total_counts = 100
    forwarder = SpeedForward(model, graph_head=model.graph_head, mode=args.mode)
    all_times = []
    _fix_randomness(args.seed)

    for _ in tqdm(range(total_counts)):
        probe_item = random.choice(imdb.probes)
        gallery_item = random.choice(imdb.roidb)
        target_items = [probe_item, gallery_item]
        if args.mode == "feat":
            target_items = [target_items[0]]

        images = [Image.open(item["path"]) for item in target_items]
        images = [extractor.transform(image) for image in images]
        images = ship_to_cuda(images, device=device)

        # ctime = forwarder(images)
        try:
            ctime = forwarder(images)
        except Exception:
            continue
        all_times.append(ctime)

    print(len(all_times))
    all_times = np.asarray(all_times)
    print("Average Inferene Time: {:2.6f} +/- {:2.6f}".format(all_times.mean(), all_times.std()))
    print(all_times.mean())
    print(all_times.std())


if __name__ == '__main__':
    main()
