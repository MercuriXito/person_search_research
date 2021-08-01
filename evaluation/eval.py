from typing import Dict, List

import os
import PIL.Image as Image
import numpy as np
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.ops as box_ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from utils import ship_to_cuda, pkl_load
from datasets import load_eval_datasets
from evaluation.context_eval import search_performance_by_sim


class Person_Search_Features_Extractor(nn.Module):
    def __init__(self, model, device=None) -> None:
        super(Person_Search_Features_Extractor, self).__init__()
        self.model = model
        self.device = device
        self.to(self.device)
        self.model.eval()
        self._init()

    def to(self, device):
        self.model.to(device)
        self.device = device

    def _init(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.min_size = 900
        self.max_size = 1500
        self.transform = T.Compose([
            T.ToTensor()
        ])
        self.infer_transform = GeneralizedRCNNTransform(
            self.min_size, self.max_size,
            self.mean, self.std
        )

    def get_query_features(self, probes: List[Dict], *args, **kwargs):
        raise NotImplementedError()

    def get_gallery_features(self, galleries: List[Dict], *args, **kwargs):
        raise NotImplementedError()


class Extractor(Person_Search_Features_Extractor):
    def get_gallery_features(self, galleries: List[Dict], nms_thresh=0.5):
        gallery_features = []
        gallery_rois = []

        for item in galleries:
            img_path = item["path"]
            image = Image.open(img_path)
            ow, oh = image.size

            image = self.transform(image)
            image, _ = self.infer_transform([image], None)
            image = image[0].to(self.device)
            w, h = image.shape[-2:]

            features, rois, scores = \
                self.model.extract_features_without_bboxes(image)

            # scale back to original size
            rw = w / ow
            rh = h / oh
            scale_fct = torch.as_tensor([[rw, rh, rw, rh]]).to(self.device)
            rois = rois * scale_fct

            # nms
            keep = box_ops.nms(rois, scores, iou_threshold=nms_thresh)
            rois = rois[keep]
            scores = scores[keep]
            features = features[keep]

            features = features.detach().cpu().numpy()
            rois = np.concatenate([
                rois.detach().cpu().numpy(),
                scores.detach().cpu().numpy().flatten().reshape(-1, 1)
            ], axis=1)

            gallery_features.append(features)
            gallery_rois.append(rois)

        return gallery_features, gallery_rois

    def get_query_features(self, probes: List[Dict], nms_thresh):
        query_features = []
        query_rois = []

        for item in probes:
            img_path = item["path"]
            boxes = item["boxes"]

            image = Image.open(img_path)
            ow, oh = image.size

            image = self.transform(image)
            target = [{"boxes": torch.as_tensor(boxes)}]
            image, target = self.infer_transform([image], target)
            image, target = ship_to_cuda(image, target)
            image, target = image[0], target[0]
            w, h = image.shape[-2:]

            features, rois, scores = \
                self.model.extract_query_features(image, target["boxes"])

            # scale back to original size
            rw = w / ow
            rh = h / oh
            scale_fct = torch.as_tensor([[rw, rh, rw, rh]]).to(self.device)
            rois = rois * scale_fct

            # nms
            keep = box_ops.nms(rois, scores, iou_threshold=nms_thresh)
            rois = rois[keep]
            scores = scores[keep]
            features = features[keep]

            features = features.detach().cpu().numpy()
            rois = np.concatenate([
                rois.detach().cpu().numpy(),
                scores.detach().cpu().numpy().flatten().reshape(-1, 1)
            ], axis=1)

            query_features.append(features)
            query_rois.append(rois)

        return query_features, query_rois

    def det_filter(self, features, boxes, det_thresh=0.5):
        num = len(features)
        new_features = []
        new_boxes = []

        for i in range(num):
            box = boxes[i]
            feats = features[i]

            scores = boxes[:, 4]
            keep = (scores > det_thresh)
            box = box[keep]
            feats = feats[keep]

            new_features.append(feats)
            new_boxes.append(box)
        return new_features, new_boxes


def evaluate(extractor, args):

    imdb = load_eval_datasets(args)
    probes = imdb.probes
    roidb = imdb.roidb

    use_data = args.use_data
    # extract features
    if len(use_data) > 0 and os.path.exists(use_data):
        data = pkl_load(use_data)
        print("load ok.")
        query_features = data["query_features"]
        gallery_features = data["gallery_features"]
        gallery_boxes = data["gallery_boxes"]
        query_boxes = data["query_boxes"]
    else:
        gallery_features, gallery_boxes = \
            extractor.get_gallery_features(roidb, nms_thresh=args.nms_thresh)
        query_features, query_boxes = \
            extractor.get_query_features(
                probes, nms_thresh=args.nms_thresh,
                use_query_ctx_boxes=args.eval_context)

    # evaluation
    det_ap, det_recall = imdb.detection_performance_calc(
        gallery_boxes, det_thresh=args.det_thresh,
        iou_thresh=args.nms_thresh,
        labeled_only=False)
    label_det_ap, label_det_recall = imdb.detection_performance_calc(
        gallery_boxes, det_thresh=args.det_thresh,
        iou_thresh=args.nms_thresh,
        labeled_only=True)

    mAP, top1, top5, top10, _ = search_performance_by_sim(
        imdb, probes, gallery_boxes, gallery_features, query_features,
        det_thresh=args.det_thresh, gallery_size=args.gallery_size,
        use_context=args.eval_context, graph_thred=args.graph_thred)

    table = PrettyTable(field_names=[
        "item", "det_ap", "det_recall", "labeled_ap", "labeled_recall",
        "mAP", "top1", "top5", "top10"])
    eval_res = [
        det_ap, det_recall, label_det_ap, label_det_recall,
        mAP, top1, top5, top10]
    format_eval_res = ["{:.8f}".format(res_item) for res_item in eval_res]
    format_eval_res = ["item"] + format_eval_res
    table.add_row(format_eval_res)
    print(table)

    # save evaluation results
    res_pkl = {
        "gallery_boxes": gallery_boxes,
        "gallery_features": gallery_features,
        "query_boxes": query_boxes,
        "query_features": query_features,
        "eval_res": eval_res,
        "eval_args": args,
    }

    return res_pkl, table.get_string()


if __name__ == '__main__':
    from evaluation.args import get_eval_argparser
    args = get_eval_argparser().parse_args()
    evaluate(None, args)
