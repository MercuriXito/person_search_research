import torch

import os
import numpy as np
import PIL.Image as Image
import torchvision.ops as box_ops

from datasets import load_eval_datasets
from evaluation.eval_defaults import build_and_load_from_dir
from evaluation.eval import FasterRCNNExtractor, evaluate, \
    GTFeatureExtractor
from utils.misc import ship_to_cuda, unpickle
from evaluation.evaluator import GraphPSEvaluator, PersonSearchEvaluator, get_context_sim
from tqdm import tqdm


# base_root = "../"
base_root = "./"


def choose_model_dataset(gallery, method):
    if gallery == "CUHK-SYSU":
        if method == "baseline":
            exp_dir = os.path.join(base_root, "exps/exps_cuhk")
            pkl_path = os.path.join(base_root, "exps/exps_cuhk/checkpoint.pth.eval.pkl")
        elif method == "cmm":
            exp_dir = os.path.join(base_root, "exps/exps_cuhk")
            pkl_path = os.path.join(base_root, "exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl")
        elif method == "acae":
            exp_dir = os.path.join(base_root, "exps/exps_acae/exps_cuhk.closs_35")
            pkl_path = os.path.join(base_root, "exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.validate.pkl")
        else:
            raise NotImplementedError(f"{method}")
    elif gallery == "PRW":
        if method == "baseline":
            exp_dir = os.path.join(base_root, "exps/exps_prw.oim")
            pkl_path = os.path.join(base_root, "exps/exps_prw.oim/checkpoint.pth.eval.pkl")
        elif method == "cmm":
            exp_dir = os.path.join(base_root, "exps/exps_prw.oim")
            pkl_path = os.path.join(base_root, "exps/exps_prw.oim/checkpoint.pth.ctx.G0.4.eval.pkl")
        elif method == "acae":
            exp_dir = os.path.join(base_root, "exps/exps_acae/exps_prw.closs_60")
            pkl_path = os.path.join(base_root, "exps/exps_acae/exps_prw.closs_60/checkpoint.pth.acae.G0.4.eval.pkl")
        else:
            raise NotImplementedError(f"{method}")
    else:
        raise NotImplementedError(f"{gallery}")
    return exp_dir, pkl_path


class FeatureExtractor(FasterRCNNExtractor):
    def get_query_features(self, probes, use_query_ctx_boxes=False, *args, **kwargs):
        query_features = []
        query_rois = []

        for item in probes:
            image, boxes = item

            image = Image.fromarray(image)
            images = [self.transform(image)]

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


def search(images, boxes, method, gallery):

    if method == "CMM":
        return search_with_cmm(images, boxes, method, gallery)
    elif method == "ACAE":
        return search_with_acae(images, boxes, method, gallery)

    exp_dir, pkl_path = choose_model_dataset(gallery, method)

    model, t_args = build_and_load_from_dir(exp_dir)
    eval_args = t_args.eval
    device = torch.device(eval_args.device)
    extractor = FeatureExtractor(model, device)
    dataset = load_eval_datasets(eval_args)

    boxes = [box.reshape(1, 4) for box in boxes]
    # form the item of query
    features, rois = extractor.get_query_features(
        list(zip(images, boxes))
    )
    features, rois = features[0], rois[0]

    # load features pickle
    data = unpickle(pkl_path)
    gallery_features = data["gallery_features"]
    gallery_boxes = data["gallery_boxes"]

    num_persons = np.asarray([0] + [len(feat) for feat in gallery_features])
    inc_num_persons = np.cumsum(num_persons)

    # cosine similarity
    all_gfeats = np.concatenate(gallery_features, axis=0)
    sim = np.matmul(features, all_gfeats.T).flatten()
    topk_indices = np.argsort(sim)[::-1][:10]

    topk_img_indices = np.sum(
        topk_indices.reshape(-1, 1) >= inc_num_persons.reshape(1, -1),
        axis=1) - 1
    topk_ind_in_img = topk_indices - inc_num_persons[topk_img_indices]
    topk_img_indices = topk_img_indices.tolist()
    topk_ind_in_img = topk_ind_in_img.tolist()

    results = []
    for img_idx, idx_in_img in zip(topk_img_indices, topk_ind_in_img):
        item = dataset.roidb[img_idx]
        box = gallery_boxes[img_idx][idx_in_img].tolist()[:4]
        box = [int(x) for x in box]
        gimage = np.asarray(Image.open(item["path"]))
        results.append(dict(
            image=gimage, box=box
        ))
    return results


def search_with_cmm(images, boxes, method, gallery):

    exp_dir, pkl_path = choose_model_dataset(gallery, method)

    model, t_args = build_and_load_from_dir(exp_dir)
    eval_args = t_args.eval
    device = torch.device(eval_args.device)
    extractor = FeatureExtractor(model, device)
    dataset = load_eval_datasets(eval_args)

    boxes = [box.reshape(1, 4) for box in boxes]
    # form the item of query
    features, rois = extractor.get_query_features(
        list(zip(images, boxes)), use_query_ctx_boxes=True
    )
    features, rois = features[0], rois[0]

    # load features pickle
    data = unpickle(pkl_path)
    gallery_features = data["gallery_features"]
    gallery_boxes = data["gallery_boxes"]

    # similarity with cmm.
    evaluator = PersonSearchEvaluator()
    sim = []
    for gfeat in tqdm(gallery_features):
        if len(gfeat) == 0:
            continue
        gsim = evaluator.get_similarity(
            gfeat, features, True, graph_thred=eval_args.graph_thred)
        sim.append(gsim)
    sim = np.concatenate(sim, axis=0).flatten()

    # find topk ranking persons
    topk_indices = np.argsort(sim)[::-1][:10]
    num_persons = np.asarray([0] + [len(feat) for feat in gallery_features])
    inc_num_persons = np.cumsum(num_persons)
    topk_img_indices = np.sum(
        topk_indices.reshape(-1, 1) >= inc_num_persons.reshape(1, -1),
        axis=1) - 1
    topk_ind_in_img = topk_indices - inc_num_persons[topk_img_indices]
    topk_img_indices = topk_img_indices.tolist()
    topk_ind_in_img = topk_ind_in_img.tolist()

    results = []
    for img_idx, idx_in_img in zip(topk_img_indices, topk_ind_in_img):
        item = dataset.roidb[img_idx]
        box = gallery_boxes[img_idx][idx_in_img].tolist()[:4]
        box = [int(x) for x in box]
        gimage = np.asarray(Image.open(item["path"]))
        results.append(dict(
            image=gimage, box=box
        ))
    return results


def search_with_acae(images, boxes, method, gallery):
    exp_dir, pkl_path = choose_model_dataset(gallery, method)

    model, t_args = build_and_load_from_dir(exp_dir)
    eval_args = t_args.eval
    device = torch.device(eval_args.device)
    extractor = FeatureExtractor(model, device)
    dataset = load_eval_datasets(eval_args)

    boxes = [box.reshape(1, 4) for box in boxes]
    # form the item of query
    features, rois = extractor.get_query_features(
        list(zip(images, boxes)), use_query_ctx_boxes=True
    )
    features, rois = features[0], rois[0]

    # load features pickle
    data = unpickle(pkl_path)
    gallery_features = data["gallery_features"]
    gallery_boxes = data["gallery_boxes"]

    # original ranking
    all_gfeats = np.concatenate(gallery_features, axis=0)
    sim = np.matmul(features[-1], all_gfeats.T).flatten()
    topk_indices = np.argsort(sim)[::-1][:100]

    num_persons = np.asarray([0] + [len(feat) for feat in gallery_features])
    inc_num_persons = np.cumsum(num_persons)
    topk_img_indices = np.sum(
        topk_indices.reshape(-1, 1) >= inc_num_persons.reshape(1, -1),
        axis=1) - 1
    topk_ind_in_img = topk_indices - inc_num_persons[topk_img_indices]

    # refined with acae.
    lower = sim[topk_indices].min()
    lower_gsim = 1.0
    evaluator = GraphPSEvaluator(model.graph_head, device)
    gsims = []
    for idx in tqdm(topk_img_indices):
        gfeat = gallery_features[idx]
        gsim = evaluator.get_similarity(
            gfeat, features, True, graph_thred=eval_args.graph_thred)
        lower_gsim = min(gsim.min(), lower_gsim)
        gsims.append(gsim)
    for idx, idx_in_img, gsim in zip(topk_indices, topk_ind_in_img, gsims):
        gsim = gsim - lower_gsim + lower
        sim[idx] = gsim[idx_in_img]

    # find topk ranking persons
    topk_indices = np.argsort(sim)[::-1][:10]
    num_persons = np.asarray([0] + [len(feat) for feat in gallery_features])
    inc_num_persons = np.cumsum(num_persons)
    topk_img_indices = np.sum(
        topk_indices.reshape(-1, 1) >= inc_num_persons.reshape(1, -1),
        axis=1) - 1
    topk_ind_in_img = topk_indices - inc_num_persons[topk_img_indices]
    topk_img_indices = topk_img_indices.tolist()
    topk_ind_in_img = topk_ind_in_img.tolist()

    results = []
    for img_idx, idx_in_img in zip(topk_img_indices, topk_ind_in_img):
        item = dataset.roidb[img_idx]
        box = gallery_boxes[img_idx][idx_in_img].tolist()[:4]
        box = [int(x) for x in box]
        gimage = np.asarray(Image.open(item["path"]))
        results.append(dict(
            image=gimage, box=box
        ))
    return results

