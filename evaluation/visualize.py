import cv2
import torch
import random
import PIL.Image as Image
import numpy as np

import os
from tqdm import tqdm

from utils.vis import draw_boxes_text
from datasets import load_eval_datasets
from evaluation.eval import FasterRCNNExtractor
from evaluation.eval_defaults import build_and_load_from_dir


def _fix_randomness(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def vis_det_res(base_root):
    """ visualize the detection ability.
    """

    model, args = build_and_load_from_dir(base_root)
    dataset = load_eval_datasets(args.eval)
    eval_args = args.eval
    device = torch.device(eval_args.device)
    extractor = FasterRCNNExtractor(model, device)

    vis_root = os.path.join(base_root, "vis")
    vis_det_root = os.path.join(vis_root, "vis_det")
    os.makedirs(vis_det_root, exist_ok=True)

    # sample data
    roidb = dataset.gt_roidb()
    num_samples = 10
    # sample_indices = np.random.choice(
    #     np.arange(len(roidb)).astype(np.int), size=num_samples
    # ).tolist()
    sample_indices = list(range(num_samples))  # fixed

    # forward
    for ind in tqdm(sample_indices):
        sample = roidb[ind]
        im_name, path, gt_boxes = sample["im_name"], sample["path"], sample["boxes"]

        sample = [sample]
        _, rois = extractor.get_gallery_features(sample)
        rois = rois[0]

        # process name
        ext = im_name.split(".")[-1]
        former_name = ".".join(im_name.split(".")[:-1])

        image = np.asarray(Image.open(path))
        # draw ground truth boxes
        gt_image = draw_boxes_text(image, gt_boxes)
        gt_save_path = os.path.join(vis_det_root, f"{former_name}_gt.{ext}")
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(gt_save_path, gt_image)

        # draw extracted boxes
        boxes = rois[:, :4]
        scores = [f"{score:2.2f}" for score in rois[:, 4].tolist()]
        c_scores = [f"{score:2.2f}" for score in rois[:, -1].tolist()]
        sscores = [f"{c}_{cs}" for c, cs in zip(scores, c_scores)]
        res_image = draw_boxes_text(image, boxes, str_texts=sscores)
        res_save_path = os.path.join(vis_det_root, f"{former_name}_res.{ext}")
        res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(res_save_path, res_image)


if __name__ == '__main__':
    _fix_randomness()
    # vis_det_res("exps/exps_det/exps_cuhk.retinanet.fpn_reid_head.sample")
    # vis_det_res("exps/exps_det/exps_cuhk.fcos.loss_cls_5")
    # vis_det_res("exps/exps_det/exps_cuhk.fcos")
    # vis_det_res("exps/exps_det/exps_cuhk.retinanet.loss_weights")
    # vis_det_res("exps/exps_det/exps_cuhk.fcos.loss_cls_5.center")
    # vis_det_res("exps/exps_det/exps_cuhk.retinanet.loss_weights.iter")
    # vis_det_res("exps/exps_det/exps_cuhk.fcos.loss_cls_5.center.iter")
    vis_det_res("exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head")
