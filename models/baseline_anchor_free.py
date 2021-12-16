import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection._utils import BoxCoder
import torchvision.ops.boxes as box_ops
from models.backbone import build_fpn_backbone


# ---------------------- supplementary functions -------------------
def _area(boxes):
    areas = (boxes[..., 2] - boxes[..., 0]) * \
        (boxes[..., 3] - boxes[..., 1])
    return areas


def encode_centerness(regression):
    assert regression.ndim == 2

    left = regression[..., 0]
    top = regression[..., 1]
    right = regression[..., 2]
    bottom = regression[..., 3]

    width_offset = torch.stack([left, right], dim=-1)
    height_offset = torch.stack([top, bottom], dim=-1)

    width_offset = width_offset.min(dim=-1)[0] / width_offset.max(dim=-1)[0]
    height_offset = height_offset.min(dim=-1)[0] / height_offset.max(dim=-1)[0]

    target_centerness = torch.sqrt(width_offset * height_offset)
    return target_centerness


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


def sigmoid_focal_loss(
        inputs, targets, reduction="sum",
        gamma: float = 2, alpha: float = 0.25):
    pred = torch.sigmoid(inputs)
    labels = targets.type_as(inputs)

    weights = alpha * (1 - pred) ** gamma * labels + \
        (1 - alpha) * pred ** gamma * (1 - labels)

    loss = weights * F.binary_cross_entropy_with_logits(
        inputs, labels, reduction="none")

    if reduction == "sum":
        return loss.sum()
    else:
        return loss


class FCOSBoxCoder(BoxCoder):
    """ Box Encoder and Decoder as defined in FCOS method.
    Different from anchor based methods, boxes are decoded with points
    and relative codes. Boxes in x1y1x2y2 format. Codes in ltrb format.
    """
    def __init__(self, weights) -> None:
        super().__init__(weights, bbox_xform_clip=None)

    def encode(self, reference_points, proposals, image_shapes):
        assert isinstance(reference_points, (list, tuple))
        assert isinstance(proposals, (list, tuple))

        boxes_per_image = [len(b) for b in reference_points]
        reference_boxes = torch.cat(reference_points, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals, image_shapes)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, proposals, reference_points, image_shapes):

        image_height, image_width = image_shapes

        x1 = proposals[:, 0]
        y1 = proposals[:, 1]
        x2 = proposals[:, 2]
        y2 = proposals[:, 3]

        cx, cy = reference_points[:, 0], reference_points[:, 1]

        wl, wt, wr, wb = self.weights
        left = (cx - x1) * wl / image_width
        top = (cy - y1) * wt / image_height
        right = (x2 - cx) * wr / image_width
        bottom = (y2 - cy) * wb / image_height
        return torch.stack([left, top, right, bottom], dim=1)

    def decode(self, rel_codes, points, image_shapes):
        assert isinstance(rel_codes, (list, tuple))
        assert isinstance(points, (list, tuple))

        boxes = []
        for red_codes_p, points_p in zip(rel_codes, points):
            boxes.append(
                self.decode_single(red_codes_p, points_p, image_shapes))
        boxes = torch.cat(boxes, dim=0)
        return boxes

    def decode_single(self, rel_codes, points, image_shapes):

        image_height, image_width = image_shapes

        points = points.to(rel_codes.dtype)
        x, y = points[:, 0][..., None], points[:, 1][..., None]

        wl, wt, wr, wb = self.weights
        left = rel_codes[:, 0::4] / wl * image_width
        top = rel_codes[:, 1::4] / wt * image_height
        right = rel_codes[:, 2::4] / wr * image_width
        bottom = rel_codes[:, 3::4] / wb * image_height

        pred_boxes1 = x - left
        pred_boxes2 = y - top
        pred_boxes3 = x + right
        pred_boxes4 = y + bottom

        return torch.cat([pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4], dim=-1)


def _reshape(x):
    # x[N K H W] -> [N HW K]
    assert x.ndim == 4
    N, _, H, W = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(N, H*W, -1)
    return x


# ---------------------- models -------------------
class FCOSClassificationHead(nn.Module):
    """
    A classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(
            self, in_channels, num_classes,
            use_centerness=True, prior_probability=0.01):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        # self.use_centerness = use_centerness
        # TODO: enable/disable centerness
        self.use_centerness = True
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.centerness.weight, std=0.01)
        torch.nn.init.constant_(self.centerness.bias, 0)

        self.num_classes = num_classes
        self.box_coder = FCOSBoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_cls_loss(self, targets, head_outputs, matched_idxs):
        losses = []

        cls_logits = head_outputs['cls_logits']
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                # HACK: compatible with labels annotation.
                targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]] - 1
            ] = 1.0

            # find indices for which anchors should be ignored
            # valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            # losses.append(sigmoid_focal_loss(
            #     cls_logits_per_image[valid_idxs_per_image],
            #     gt_classes_target[valid_idxs_per_image],
            #     reduction='sum',
            # ) / max(1, num_foreground))

            # TODO: decide which kind of samples to omit.
            losses.append(sigmoid_focal_loss(
                cls_logits_per_image,
                gt_classes_target,
                reduction='sum',
            ) / max(1, num_foreground))

        return _sum(losses) / len(targets)

    def compute_centerness_loss(self, targets, head_outputs, points, matched_idxs, image_shapes):
        losses = []

        centerness = head_outputs["centerness"]
        for targets_per_image, centerness_per_image, matched_idxs_per_image, points_per_image, image_shape in \
                zip(targets, centerness, matched_idxs, points, image_shapes):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            points_per_image = points_per_image[foreground_idxs_per_image, :]
            centerness_per_image = centerness_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(
                matched_gt_boxes_per_image, points_per_image, image_shape)
            target_centerness = encode_centerness(target_regression).view_as(centerness_per_image)

            losses.append(
                F.binary_cross_entropy_with_logits(
                    centerness_per_image,
                    target_centerness,
                    reduction="mean")
            )

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []
        all_centerness = []

        for features in x:
            features = self.conv(features)
            cls_logits = self.cls_logits(features)
            cls_logits = _reshape(cls_logits)
            all_cls_logits.append(cls_logits)

            if self.use_centerness:
                centerness = self.centerness(features)
                centerness = _reshape(centerness)
                all_centerness.append(centerness)

        all_cls_logits = torch.cat(all_cls_logits, dim=1)

        if self.use_centerness:
            all_centerness = torch.cat(all_centerness, dim=1)
        return all_cls_logits, all_centerness


class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
    """

    def __init__(self, in_channels):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # Differ from RetinaNet1: one output for one location.
        self.bbox_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)
        self.box_coder = FCOSBoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self, targets, head_outputs, points, matched_idxs, image_shapes):
        losses = []
        bbox_regression = head_outputs['bbox_regression']

        for targets_per_image, bbox_regression_per_image, points_per_image, matched_idxs_per_image, image_shape in \
                zip(targets, bbox_regression, points, matched_idxs, image_shapes):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            points_per_image = points_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, points_per_image, image_shape)

            # compute the loss
            losses.append(torch.nn.functional.l1_loss(
                bbox_regression_per_image,
                target_regression,
                reduction='sum'
            ) / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            bbox_regression = _reshape(bbox_regression)
            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class FCOSHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classification_head = FCOSClassificationHead(in_channels, num_classes)
        self.regression_head = FCOSRegressionHead(in_channels)

    def compute_loss(self, targets, head_outputs, points, matched_idxs, image_shapes):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            'loss_classifier': self.classification_head.compute_cls_loss(targets, head_outputs, matched_idxs),
            'loss_centerness': self.classification_head.compute_centerness_loss(targets, head_outputs, points, matched_idxs, image_shapes),
            'loss_box_reg': self.regression_head.compute_loss(targets, head_outputs, points, matched_idxs, image_shapes),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        cls, center = self.classification_head(x)
        return {
            'cls_logits': cls,
            'bbox_regression': self.regression_head(x),
            'centerness': center,
        }


class AnchorFreePS(nn.Module):
    def __init__(
            self,
            backbone,
            fcos_head,
            transform,
            # other parameters
            topk_candidates=1000,
            detections_per_img=300,
            score_thresh=0.05,
            nms_thresh=0.5,
            ):
        super(AnchorFreePS, self).__init__()
        self.backbone = backbone
        self.fcos_head = fcos_head
        self.transform = transform

        self.box_coder = FCOSBoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.topk_candidates = topk_candidates
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(self, images, targets=None):
        """
        args:
            - image: List[Tensor]
            - targets: List[Dict(str, Tensor)]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features_dict = self.backbone(images.tensors)
        features = list(features_dict.values())
        head_outputs = self.fcos_head(features)

        points, feats_shapes, feats_strids = \
            self.reference_points(images, features)

        if self.training:
            loss, matched_idxs = \
                self.compute_loss(points, head_outputs, targets, images.image_sizes)
            return None, loss
        else:
            detections = self.postprocess(points, head_outputs, feats_shapes, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections, points, feats_shapes, head_outputs

    def compute_loss(self, points, head_outputs, targets, image_shapes):
        # TODO: constraint the matching of groud-truth boxes for points on
        # different level of features
        device = points.device

        all_matched_idxs = []
        # match between points and ground truth boxes
        for target_per_img, points_per_img in zip(targets, points):
            boxes = target_per_img["boxes"].unsqueeze(dim=0)
            points_per_img = points_per_img.unsqueeze(dim=1)

            # in boxes
            left_cond = points_per_img[..., 0] >= boxes[..., 0]
            right_cond = points_per_img[..., 0] <= boxes[..., 2]
            top_cond = points_per_img[..., 1] >= boxes[..., 1]
            bottom_cond = points_per_img[..., 1] <= boxes[..., 3]

            width_cond = torch.logical_and(left_cond, right_cond)
            height_cond = torch.logical_and(top_cond, bottom_cond)
            cond = torch.logical_and(width_cond, height_cond)  # [NxK]

            matched = torch.sum(cond, dim=1) > 0

            # cost based on the areas of boxes
            box_areas = _area(boxes)
            cond = cond.type(torch.int)
            cost = cond * box_areas + (1 - cond) * torch.iinfo(torch.int).max
            cost_idxs = cost.argmin(dim=1)

            matched_idxs = torch.full((len(points_per_img), ), -1, dtype=torch.long, device=device)
            matched_idxs[matched] = cost_idxs[matched]
            all_matched_idxs.append(matched_idxs)

        loss = self.fcos_head.compute_loss(
            targets, head_outputs, points, all_matched_idxs, image_shapes)
        return loss, all_matched_idxs

    def postprocess(self, points, head_outputs, feats_shapes, image_shapes):

        regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]
        centerness = head_outputs["centerness"]

        detections = []
        for points_per_image, reg_per_image, cls_per_image, center_per_image, image_shape in \
                zip(points, regression, cls_logits, centerness, image_shapes):

            points_list = torch.split(points_per_image, feats_shapes)
            regression_list = torch.split(reg_per_image, feats_shapes)
            cls_logits_list = torch.split(cls_per_image, feats_shapes)
            centerness_list = torch.split(center_per_image, feats_shapes)

            image_boxes = []
            image_scores = []
            image_labels = []

            for p_level, reg_level, cls_level, cet_level in \
                    zip(points_list, regression_list, cls_logits_list, centerness_list):

                num_classes = cls_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(cls_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = topk_idxs // num_classes
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    reg_level[anchor_idxs], p_level[anchor_idxs], image_shape)
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })

        return detections

    def reference_points(self, images, features):
        feats_shapes = [s.shape[-2] * s.shape[-1] for s in features]
        image_height, image_width = images.tensors.shape[-2:]
        feats_strides = []
        points = []
        for x in features:
            H, W = x.shape[-2:]
            stride_height, stride_width = image_height // H, image_width // W
            strides = torch.tensor([stride_width, stride_height])
            feats_strides.append(strides)

            cx = torch.arange(0, W).to(x) * stride_width + stride_width // 2
            cy = torch.arange(0, H).to(x) * stride_height + stride_height // 2

            cy, cx = torch.meshgrid(cy, cx)
            points_feat = torch.stack([cx, cy], dim=-1)
            points_feat = points_feat.flatten(start_dim=0, end_dim=1)
            points.append(points_feat)

        points = torch.cat(points).unsqueeze(dim=0)
        points = points.expand(2, -1, -1)
        return points, feats_shapes, feats_strides


def build_anchor_free_based_models(args):

    min_size = 800
    max_size = 1333

    # build tranform
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    # build fpn backbone
    backbone = build_fpn_backbone(
            args.model.backbone.name,
            args.model.backbone.pretrained,
            args.model.backbone.norm_layer
    )

    num_classes = 1
    fcos_head = FCOSHead(256, num_classes=num_classes)

    model = AnchorFreePS(backbone, fcos_head, transform)
    return model


if __name__ == '__main__':
    from datasets.transforms import ToTensor
    from configs.faster_rcnn_default_configs import get_default_cfg
    from utils.misc import ship_to_cuda
    from datasets import build_trainset

    args = get_default_cfg()
    device = "cuda"
    # device = "cpu"
    device = torch.device(device)

    root = "data/cuhk-sysu"
    transforms = ToTensor()

    # dataset = CUHK_SYSU(root, transforms, "train")
    dataset = build_trainset("cuhk-sysu", root, False)
    image1, target1 = dataset[0]
    image2, target2 = dataset[1]
    images = [image1, image2]
    targets = [target1, target2]
    images, targets = ship_to_cuda(images, targets, device)

    model = build_anchor_free_based_models(args)
    model.load_state_dict(
        torch.load("exps/exps_det/exps_fcos.det/checkpoint.pth", map_location="cpu")["model"]
    )
    model.to(device)
    model.eval()
    # model.train()

    with torch.no_grad():
        outputs = model(images, targets)

    from IPython import embed
    embed()

    # import cv2
    # from utils.vis import draw_boxes_text

    # det_thresh = 0.5

    # outputs = outputs[0]
    # for image, target, outs in zip(images, targets, outputs):
    #     boxes = outs["boxes"]
    #     scores = outs["scores"]

    #     keep = scores > det_thresh
    #     boxes = boxes[:5]

    #     dimg = draw_boxes_text(image, boxes)
    #     print(cv2.imwrite(f"exps/vis/fcos_det_{target['im_name']}", dimg))

    if True:
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np

        detections, points, feats_shapes, head_outputs = outputs

        for idx, (dets, pts, shape, image, target) in \
                enumerate(zip(detections, points, feats_shapes, images, targets)):

            ih, iw = image.shape[-2:]
            boxes = dets["boxes"]
            scores = dets["scores"]

            lvl_points = pts.split(shape)[0]
            cls_scores = head_outputs["cls_logits"][0]
            lvl_cls_scores = cls_scores.split(shape)[0]
            lvl_cls_scores = torch.sigmoid(lvl_cls_scores)

            stride = 2 ** (0 + 3)
            lvl_points = (lvl_points - stride // 2) // stride
            lvl_points = lvl_points.type(torch.long)
            w, h = lvl_points[-1]

            lvl_cls_scores = lvl_cls_scores.reshape(h + 1, w + 1)
            lvl_cls_scores = lvl_cls_scores.detach().cpu().numpy()
            lvl_cls_scores = np.resize(lvl_cls_scores, (ih, iw))

            scores = lvl_cls_scores
            scores = (scores - scores.min()) / (scores.max() - scores.min())

            # prcoess image
            cimage = image.detach().cpu().numpy().transpose(1, 2, 0)
            cimage = np.clip(cimage, 0, 1) * 255.0

            # process heatmap
            scores = np.clip(scores, 0, 1) * 255.0
            scores = scores.astype(np.uint8)
            scores = cv2.applyColorMap(scores, cv2.COLORMAP_JET)

            cimage = cimage * 0.6 + scores * 0.4
            cimage = cimage.astype(np.uint8)

            # RGB2BGR
            cimage = cv2.cvtColor(cimage, cv2.COLOR_RGB2BGR)

            print(cv2.imwrite(f"exps/vis/fcos_{target['im_name']}", cimage))
