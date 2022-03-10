from typing import Dict, List

import os
import PIL.Image as Image
from prettytable import PrettyTable
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.ops as box_ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from utils.misc import ship_to_cuda, unpickle
from datasets import load_eval_datasets
from evaluation.evaluator import PersonSearchEvaluator


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

    @staticmethod
    def rescale_boxes(boxes, sizes, target_sizes):
        """
        args:
            - boxes: List[Tensor], in x1y1x2y2 format
            - sizes: List[(h, w)]
            - target_sizes: List[(h, w)]
        """
        ratios = [
            [ts[0] / s[0], ts[1] / s[1]]
            for s, ts in zip(sizes, target_sizes)
        ]
        new_boxes = []
        for i, box in enumerate(boxes):
            hr, wr = ratios[i]
            scale_ratio = torch.as_tensor([wr, hr, wr, hr])
            scale_ratio = scale_ratio.view(1, 4).to(box.device)
            box = box.view(-1, 4) * scale_ratio
            new_boxes.append(box)
        return new_boxes


class FasterRCNNExtractor(Person_Search_Features_Extractor):
    """ Default Feature Extractor for faster-rcnn based models.
    """
    def __init__(self, model, device) -> None:
        super().__init__(model, device=device)

    def get_gallery_features(self, galleries, *args, **kwargs):
        gallery_features = []
        gallery_rois = []

        for item in tqdm(galleries):
            img_path = item["path"]
            image = Image.open(img_path)
            image = [self.transform(image)]
            image = ship_to_cuda(image, device=self.device)

            outputs = self.model.extract_features_without_boxes(image)
            outputs = outputs[0]

            boxes, features, scores = \
                outputs["boxes"], outputs["embeddings"], outputs["scores"]

            scores = scores.view(-1, 1)
            rois = torch.cat([boxes, scores], dim=1)
            rois = rois.detach().cpu().numpy()
            features = features.detach().cpu().numpy()

            if "centerness" in outputs:
                centerness = outputs["centerness"]
                centerness = centerness.view(-1, 1).detach().cpu().numpy()
                import numpy as np
                rois = np.concatenate([rois, centerness], axis=1)

            gallery_features.append(features)
            gallery_rois.append(rois)

        return gallery_features, gallery_rois

    def get_query_features(self, probes, use_query_ctx_boxes=False, *args, **kwargs):
        query_features = []
        query_rois = []

        for item in tqdm(probes):
            img_path = item["path"]
            image = Image.open(img_path)
            images = [self.transform(image)]

            boxes = item["boxes"]
            scores = torch.as_tensor([1])
            targets = [dict(boxes=boxes, scores=scores)]
            images, targets = ship_to_cuda(images, targets, self.device)

            if use_query_ctx_boxes:
                # extract contextual query boxes
                outputs = self.model.extract_features_without_boxes(images)
                o_boxes = [o["boxes"] for o in outputs]
                o_scores = [o["scores"] for o in outputs]
                num_imgs = len(o_boxes)

                all_boxes, all_scores = [], []
                for i in range(num_imgs):
                    box, score = o_boxes[i], o_scores[i]
                    gt_qbox, gt_score = targets[i]["boxes"], targets[i]["scores"]

                    all_box = torch.cat([box, gt_qbox])
                    all_score = torch.cat([score, gt_score])
                    keep = box_ops.nms(all_box, all_score, iou_threshold=0.4)
                    all_box = all_box[keep]
                    all_score = all_score[keep]

                    assert all_score[0] == 1
                    # move the gt boxes to the last one
                    all_box = torch.cat([all_box[1:], all_box[0].view(-1, 4)])
                    all_score = torch.cat([all_score[1:], all_score[0].view(1)])
                    all_boxes.append(all_box)
                    all_scores.append(all_score)

                new_targets = [
                    dict(boxes=b, scores=s)
                    for b, s in zip(all_boxes, all_scores)
                ]
            else:
                new_targets = targets

            boxes = [t["boxes"] for t in new_targets]
            scores = [t["scores"].view(-1, 1) for t in new_targets]

            # support batch_size=1 only
            boxes = boxes[0]
            scores = scores[0]

            outputs = self.model.extract_features_with_boxes(images, new_targets)
            features = outputs

            rois = torch.cat([boxes, scores], dim=1)
            rois = rois.detach().cpu().numpy()
            features = features.detach().cpu().numpy()

            query_features.append(features)
            query_rois.append(rois)

        return query_features, query_rois


class GTFeatureExtractor(FasterRCNNExtractor):
    """ Use ground truth gallery boxes to extract features, while keeping
    query as the same. This avoids effect from detection to searching.
    """
    def __init__(self, model, device=None) -> None:
        super().__init__(model, device=device)

    @torch.no_grad()
    def get_gallery_features(self, galleries, *args, **kwargs):
        gallery_features = []
        gallery_rois = []

        for item in tqdm(galleries):
            img_path = item["path"]
            image = Image.open(img_path)

            scores = torch.tensor([1] * len(item["boxes"])).reshape(-1, 1)
            boxes = torch.tensor(item["boxes"])
            targets = [dict(boxes=item["boxes"], scores=scores)]
            image = [self.transform(image)]
            image, targets = ship_to_cuda(image, targets, device=self.device)
            features = self.model.extract_features_with_boxes(image, targets)

            scores = scores.view(-1, 1)
            rois = torch.cat([boxes, scores], dim=1)
            rois = rois.detach().cpu().numpy()
            features = features.detach().cpu().numpy()

            gallery_features.append(features)
            gallery_rois.append(rois)

        return gallery_features, gallery_rois


class GTQueryFeatureExtractor(GTFeatureExtractor):
    """ Extractor features for ground truth query target and its surrouding
    persons. This annotation is only availiable in PRW in
    `prw.load_probes_with_ctx`.
    """
    @torch.no_grad()
    def get_query_features(
            self, probes, use_query_ctx_boxes=False, *args, **kwargs):
        return self.get_gallery_features(probes)


def evaluate(extractor, args, /, imdb=None, ps_evaluator=None, res_pkl=None):
    if imdb is None:
        imdb = load_eval_datasets(args)
    probes = imdb.probes
    roidb = imdb.roidb
    if ps_evaluator is None:
        ps_evaluator = PersonSearchEvaluator(args.dataset_file)

    use_data = args.use_data
    # extract features
    if res_pkl is not None:
        print("Using res pickle.")
        query_features = res_pkl["query_features"]
        gallery_features = res_pkl["gallery_features"]
        gallery_boxes = res_pkl["gallery_boxes"]
        query_boxes = res_pkl["query_boxes"]
    elif use_data is not None and len(use_data) > 0 and os.path.exists(use_data):
        data = unpickle(use_data)
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

        # data = unpickle("test_features.pkl")
        # gallery_features = data["gallery_features"]
        # gallery_boxes = data["gallery_boxes"]
        # query_features = data["query_features"]
        # query_boxes = data["query_boxes"]

    # evaluation
    det_ap, det_recall = imdb.detection_performance_calc(
        gallery_boxes, det_thresh=args.det_thresh,
        iou_thresh=args.nms_thresh,
        labeled_only=False)
    label_det_ap, label_det_recall = imdb.detection_performance_calc(
        gallery_boxes, det_thresh=args.det_thresh,
        iou_thresh=args.nms_thresh,
        labeled_only=True)

    mAP, top1, top5, top10, ret = ps_evaluator.eval_search(
        imdb, probes, gallery_boxes, gallery_features, query_features,
        det_thresh=args.det_thresh, gallery_size=args.gallery_size,
        use_context=args.eval_context, graph_thred=args.graph_thred)

    file_names = [
        "item", "det_ap", "det_recall", "labeled_ap", "labeled_recall",
        "mAP", "top1", "top5", "top10"]
    table = PrettyTable(field_names=file_names)
    eval_res = [
        "item", det_ap, det_recall, label_det_ap, label_det_recall,
        mAP, top1, top5, top10]
    formats = ["{}"] + ["{:8f}"] * 8
    format_eval_res = [
        formats[i].format(res_item)
        for i, res_item in enumerate(eval_res)]
    table.add_row(format_eval_res)
    print(table)

    # save evaluation results
    res_pkl = {
        "gallery_boxes": gallery_boxes,
        "gallery_features": gallery_features,
        "query_boxes": query_boxes,
        "query_features": query_features,
        "eval_res": dict([(k, v) for k, v in zip(file_names, eval_res)]),
        "eval_args": args,
        "eval_rank": ret,
    }

    return res_pkl, table.get_string()


if __name__ == '__main__':
    pass
