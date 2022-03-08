"""
Person Search model based on RetinaNet.
"""
import math
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List, Tuple, Optional
import torchvision
from torchvision import ops
import torchvision.ops.boxes as box_ops

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.poolers import MultiScaleRoIAlign
from models.losses import OIMLoss
from models.backbone import build_fpn_backbone, \
    build_faster_rcnn_based_multi_scale_backbone
from models.reid_head import ReIDEmbeddingHead


class PSRoIHead(nn.Module):
    def __init__(
            self,
            box_roi_pool,
            box_head,
            reid_embed_head,
            oim_loss,
            use_k_sampling=True,
            k=16):
        """
        mode: "gt": only use ground truth boxes for training,
              "sampling": use gt boxes and sampled generated boxes.
        """
        super().__init__()
        self.box_head = box_head
        self.reid_embed_head = reid_embed_head
        self.oim_loss = oim_loss
        self.use_k_sampling = use_k_sampling
        self.k = k
        self.box_roi_pool = box_roi_pool

    def forward(self,
                features: dict,
                proposals: list,
                image_shapes,
                targets=None,
                matched_idxs=None,
                *args, **kwargs):

        if self.training:
            assert matched_idxs is not None
            features_shape = [list(feats.shape) for feats in features.values()]
            # feats_level_inds is required if using level_roi_pooling.
            proposals, labels, pid_labels, feats_level_inds = \
                self.select_training_samples(proposals, targets, features_shape, matched_idxs)

        num_imgs = len(proposals)
        # roi_features = self.level_roi_pooling(features, proposals, feats_level_inds)
        roi_features = self.box_roi_pool(features, proposals, image_shapes)
        roi_features = self.box_head(roi_features)
        embeddings, norms = self.reid_embed_head(roi_features)

        results = []
        losses = {}
        if self.training:
            loss_oim = self.oim_loss(embeddings, pid_labels)
            losses = {"loss_oim": loss_oim}
        else:
            num_persons_per_images = [len(proposal) for proposal in proposals]
            embedding_list = embeddings.split(num_persons_per_images)
            norms_list = norms.split(num_persons_per_images)
            results = []
            for idx in range(num_imgs):
                results.append({
                    "embeddings": embedding_list[idx],
                    "norm": norms_list[idx]
                })
        return results, losses

    def level_roi_pooling(self, features, proposals, feats_level_inds):
        """ ROIPooling/RoIAlign based on multi-scale features where `feats_level_inds`
        provides the exact level of features which each box is designated to apply on.
        (Unlike MultiScaleROIAlign, in which `feats_level_inds` is deducted from
        the size of boxes).
        """
        num_imgs = len(proposals)
        roi_features = [[] for _ in range(num_imgs)]
        roi_inds = [[] for _ in range(num_imgs)]

        for feat_idx, (_, feature) in enumerate(features.items()):

            boxes_per_level = []
            for img_idx, (boxes_per_img, linds_per_img) in \
                    enumerate(zip(proposals, feats_level_inds)):
                inds = torch.where(linds_per_img == feat_idx)[0]
                roi_inds[img_idx].append(inds)
                boxes_per_level.append(boxes_per_img[inds, :])

            num_boxes_per_img = [len(boxes) for boxes in boxes_per_level]
            if sum(num_boxes_per_img) == 0:
                continue

            roi_feats = torchvision.ops.roi_align(
                feature, boxes_per_level,
                output_size=14,
                spatial_scale=float(1/(2**(feat_idx+3))),  # HACK: start from C2 feature
                sampling_ratio=2,
            )

            roi_feats = roi_feats.split(num_boxes_per_img)
            for idx, roi_feat in enumerate(roi_feats):
                roi_features[idx].append(roi_feat)

        roi_inds = [torch.cat(inds) for inds in roi_inds]
        roi_features = [torch.cat(feat) for feat in roi_features]

        for i, (rfeats, rinds) in enumerate(zip(roi_features, roi_inds)):
            zero = torch.zeros_like(rfeats)
            zero[rinds] = rfeats
            roi_features[i] = zero
        roi_features = torch.cat(roi_features)

        return roi_features

    def k_sampling(self, labels):
        sample_inds = []
        for img_in_labels in labels:
            ulabels = torch.unique(img_in_labels)
            sample_in_imgs = []
            for label in ulabels:
                inds = torch.where(img_in_labels == label)[0].tolist()
                sinds = random.choices(inds, k=self.k)
                sample_in_imgs.extend(sinds)
            sample_in_imgs = torch.tensor(sample_in_imgs).to(ulabels)
            sample_inds.append(sample_in_imgs)
        return sample_inds

    def select_gt_training_samples(
            self, proposals, targets, features_shape, matched_idxs):
        """ Only sample ground-truth boxes, suggested in DMR-Net.
        """
        boxes = [target["boxes"] for target in targets]
        labels = [target["labels"]for target in targets]
        pid_labels = [target["pid_labels"] for target in targets]
        feats_level = []
        return boxes, labels, pid_labels, feats_level

    def select_training_samples(
            self, proposals, targets, features_shape, matched_idxs):
        """ K-sampling for each instance in ground truth.
            However, risk running out of memory.
            注意这种采样是 gt 中的每个 instance 都采样 K 个（包括每个 unlabeled）
            而不是所有的 unlabeled 采样 K 个
        """
        device = proposals[0].device

        spatial_sizes = [x[-1] * x[-2] for x in features_shape]
        num_anchors_per_location = proposals[0].shape[0] // sum(spatial_sizes)
        split_sizes = [s * num_anchors_per_location for s in spatial_sizes]
        start_points = [0]
        for idx in range(len(split_sizes) - 1):
            start_points.append(start_points[-1] + split_sizes[idx])
        start_points = torch.tensor(start_points).view(1, -1).to(device)

        # K-sampling for each instance
        sampled_inds = self.k_sampling(matched_idxs)

        boxes, labels, pid_labels = [], [], []
        feats_level = []  # level of features of each sampled boxes.

        for matched_idxs_per_image, boxes_per_image, targets_per_image, sample_inds_img in \
                zip(matched_idxs, proposals, targets, sampled_inds):

            num_boxes = len(sample_inds_img)
            boxes_per_image = boxes_per_image[sample_inds_img, :]
            fg_idxs_in_sampled = torch.where(
                matched_idxs_per_image[sample_inds_img] >= 0)
            fg_idxs = sample_inds_img[fg_idxs_in_sampled]

            # sampled labels
            matched_labels = torch.zeros((num_boxes, ), dtype=torch.long).to(device)
            matched_labels[fg_idxs_in_sampled] = \
                targets_per_image["labels"][matched_idxs_per_image[fg_idxs]]

            # sampled person labels
            matched_pid_labels = torch.zeros((num_boxes, ), dtype=torch.long).to(device)
            matched_pid_labels[fg_idxs_in_sampled] = \
                targets_per_image["pid_labels"][matched_idxs_per_image[fg_idxs]]

            labels.append(matched_labels)
            pid_labels.append(matched_pid_labels)
            boxes.append(boxes_per_image)

            # feature level of each
            lvl_inds = torch.sum(
                sample_inds_img.view(-1, 1) >= start_points, dim=1)
            lvl_inds = lvl_inds - 1
            lvl_inds = lvl_inds.type(torch.int)
            feats_level.append(lvl_inds)

        # append ground-truth
        # boxes = self.add_gt_proposals(boxes, targets)
        # labels = self.add_gt_labels(labels, targets)
        # pid_labels = self.add_gt_pid_labels(pid_labels, targets)
        # feats_level = self.add_gt_feats_level(feats_level, targets)

        return boxes, labels, pid_labels, feats_level

    def select_training_samples_test(
            self, proposals, targets, features_shape, matched_idxs):
        """ K-sampling for each unique pid labels in ground truth, excluding
            background samples.
        """
        device = proposals[0].device

        spatial_sizes = [x[-1] * x[-2] for x in features_shape]
        num_anchors_per_location = proposals[0].shape[0] // sum(spatial_sizes)
        split_sizes = [s * num_anchors_per_location for s in spatial_sizes]
        start_points = [0]
        for idx in range(len(split_sizes) - 1):
            start_points.append(start_points[-1] + split_sizes[idx])
        start_points = torch.tensor(start_points).view(1, -1).to(device)

        boxes, labels, pid_labels = [], [], []
        feats_level = []  # level of features of each sampled boxes.

        for matched_idxs_per_image, boxes_per_image, targets_per_image in \
                zip(matched_idxs, proposals, targets):

            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]

            boxes_per_image = boxes_per_image[foreground_idxs_per_image, :]
            inds = matched_idxs_per_image[foreground_idxs_per_image]
            matched_labels = targets_per_image["labels"][inds]
            matched_pid_labels = targets_per_image["pid_labels"][inds]

            labels.append(matched_labels)
            pid_labels.append(matched_pid_labels)
            boxes.append(boxes_per_image)

            lvl_inds = torch.sum(
                foreground_idxs_per_image.view(-1, 1) >= start_points, dim=1)
            lvl_inds = lvl_inds - 1
            lvl_inds = lvl_inds.type(torch.int)
            feats_level.append(lvl_inds)

        # K-sampling
        samples_inds = self.k_sampling(pid_labels)
        for idx in range(len(proposals)):
            inds = samples_inds[idx]
            boxes[idx] = boxes[idx][inds]
            labels[idx] = labels[idx][inds]
            pid_labels[idx] = pid_labels[idx][inds]
            feats_level[idx] = feats_level[idx][inds]

        # NOTE: this sampling methods lacks negative background samples

        # append ground-truth
        # boxes = self.add_gt_proposals(boxes, targets)
        # labels = self.add_gt_labels(labels, targets)
        # pid_labels = self.add_gt_pid_labels(pid_labels, targets)
        # feats_level = self.add_gt_feats_level(feats_level, targets)

        return boxes, labels, pid_labels, feats_level

    def add_gt_proposals(self, boxes, targets):
        return [
            torch.cat([b, torch.tensor(target["boxes"]).to(b)], dim=0)
            for b, target in zip(boxes, targets)
        ]

    def add_gt_labels(self, labels, targets):
        return [
            torch.cat([b, torch.tensor(target["labels"]).to(b)], dim=0)
            for b, target in zip(labels, targets)
        ]

    def add_gt_pid_labels(self, pid_labels, targets):
        return [
            torch.cat([b, torch.tensor(target["pid_labels"]).to(b)], dim=0)
            for b, target in zip(pid_labels, targets)
        ]

    def add_gt_feats_level(self, feats_level, targets):
        gt_feats_level = []
        for target in targets:
            boxes = target["boxes"]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # TODO: 2. deduct features_level of ground-truth boxes
        raise NotImplementedError("")


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


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            'loss_classifier': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'loss_box_reg': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
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

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
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
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction='sum',
            ) / max(1, num_foreground))

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs['bbox_regression']

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

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

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(self, backbone, num_classes,
                 transform,
                 # ps roi head
                 ps_roi_head=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone
        self.ps_roi_head = ps_roi_head

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.transform = transform

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                               device=anchors_per_image.device))
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        loss = self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
        return loss, matched_idxs

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs['cls_logits']
        box_regression = head_outputs['bbox_regression']

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in \
                    zip(box_regression_per_image, logits_per_image, anchors_per_image):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = topk_idxs // num_classes
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
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

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features_list = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features_list)

        # create the set of anchors
        anchors = self.anchor_generator(images, features_list)

        # assert isinstance(self.ps_roi_head, PSRoIHead)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            # compute the losses
            detection_losses, matched_idxs = self.compute_loss(
                targets, head_outputs, anchors)
            proposals = self.box_coder.decode(
                head_outputs["bbox_regression"], anchors)
            proposals = proposals.view(len(anchors), -1, 4).unbind(dim=0)
            ps_outs, ps_loss = self.ps_roi_head.forward(
                features, proposals, images.image_sizes,
                targets, matched_idxs=matched_idxs, images=images)
            losses.update(detection_losses)
            losses.update(ps_loss)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features_list]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs['cls_logits'].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)

            # append outputs from roi head
            proposals = [det["boxes"] for det in detections]
            # TODO: 3. features_levels in inference
            ps_outs, _ = self.ps_roi_head.forward(features, proposals, images.image_sizes)
            for i in range(len(detections)):
                detections[i].update(ps_outs[i])

            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            matched_idxs = None

        return detections, losses

    @torch.no_grad()
    def extract_features_with_boxes(self, images, targets, feature_norm=True):
        """ extract features with boxes.
        """
        images, targets = self.transform(images, targets)
        features_dict = self.backbone(images.tensors)

        proposals = [target["boxes"] for target in targets]
        # assert isinstance(self.ps_roi_head, PSRoIHead)
        roi_res, _ = self.ps_roi_head.forward(
            features_dict, proposals, images.image_sizes)

        embeddings = [res["embeddings"] for res in roi_res]
        norms = [res["norm"] for res in roi_res]
        if not feature_norm:
            embeddings = [
                embedding * norm
                for embedding, norm in zip(embeddings, norms)
            ]
        assert len(targets) == 1, "only support batch_size = 1."
        # NOTE: only support batch_size = 1
        return embeddings[0]

    @torch.no_grad()
    def extract_features_without_boxes(self, images):
        detections = self.forward(images)[0]
        return detections


def build_retinanet_based_models(args):

    # build backbone
    backbone = build_fpn_backbone(
            args.model.backbone.name,
            args.model.backbone.pretrained,
            args.model.backbone.norm_layer)
    num_classes = 1  # excluding the background

    # build tranform
    min_size = 800
    max_size = 1333
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    # build anchor generator
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    # build retina head for cls and reg
    head = RetinaNetHead(256, 3, num_classes)

    # model parameters
    use_multi_scale = args.model.use_multi_scale
    reid_feature_dim = args.model.reid_feature_dim
    # build person search head
    _, box_head = build_faster_rcnn_based_multi_scale_backbone(
            args.model.backbone.name,
            args.model.backbone.pretrained,
            args.model.backbone.norm_layer,
            return_res4=use_multi_scale)
    # build reid head
    representation_size = 1024
    if use_multi_scale:
        reid_head = ReIDEmbeddingHead(
            featmap_names=["feat_res4", "feat_res5"],
            in_channels=[256, representation_size],
            dim=reid_feature_dim, feature_norm=True)
    else:
        reid_head = ReIDEmbeddingHead(
            featmap_names=['feat_res5'], in_channels=[256],
            dim=reid_feature_dim, feature_norm=True)
    # build oim loss
    num_features = reid_feature_dim
    num_pids = args.loss.oim.num_pids
    num_cq_size = args.loss.oim.num_cq_size
    oim_momentum = args.loss.oim.oim_momentum
    oim_scalar = args.loss.oim.oim_scalar
    oim_loss = OIMLoss(num_features, num_pids, num_cq_size, oim_momentum, oim_scalar)

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['feat_res3', 'feat_res4', 'feat_res5'],
        output_size=7,
        sampling_ratio=2)

    ps_roi_head = PSRoIHead(
            box_roi_pool, box_head, reid_head, oim_loss)

    network = RetinaNet(
        backbone, num_classes,
        transform=transform,
        ps_roi_head=ps_roi_head,
        anchor_generator=anchor_generator,
        head=head,
        # default parameters
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5, bg_iou_thresh=0.4,
        topk_candidates=1000
    )

    return network


def search_image_indices_by_pid(dataset, pid_label):
    indices = []
    for idx in range(len(dataset)):
        plabels = dataset.record[idx]["gt_pids"]
        if pid_label in list(plabels):
            indices.append(idx)
    return indices


if __name__ == '__main__':
    from configs.faster_rcnn_default_configs import get_default_cfg
    from utils.misc import ship_to_cuda
    from datasets import build_trainset

    args = get_default_cfg()

    device = "cuda"
    # device = "cpu"
    device = torch.device(device)

    root = "data/cuhk-sysu"
    dataset = build_trainset("cuhk-sysu", root, use_transform=False)

    indices = search_image_indices_by_pid(dataset, 1101)
    images, targets = [], []

    for idx in indices[:2]:
        item = dataset[idx]
        images.append(item[0])
        targets.append(item[1])
    images, targets = ship_to_cuda(images, targets, device)

    # model
    model = build_retinanet_based_models(args)
    mkeys, ukeys = model.load_state_dict(
        torch.load(
            # "exps/exps_det/exps_retinanet.det/checkpoint.pth",
            # "exps/exps_det/exps_cuhk.retinanet/checkpoint0019.pth",
            "exps/exps_det/exps_cuhk.retinanet.fpn_reid_head.sample/checkpoint0020.pth",
            map_location="cpu"
        )["model"],
        strict=False
    )

    model.to(device)
    model.eval()
    # model.train()

    with torch.no_grad():
        outputs, _ = model(images, targets)

    from IPython import embed
    embed()

    if False:
        import os
        import cv2
        from utils.vis import draw_boxes_text

        dirname = os.path.join("exps", "vis", "retinanet")
        os.makedirs(dirname, exist_ok=True)

        for image, target, outs in zip(images, targets, outputs):
            boxes = outs["boxes"]
            scores = outs["scores"]

            dimg = draw_boxes_text(image, boxes)
            print(cv2.imwrite(os.path.join(dirname, f"det_{target['im_name']}"), dimg))
