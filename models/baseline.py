import torch
import torch.nn.functional as F

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops

from models.losses import OIMLoss
from models.reid_head import ReIDEmbeddingHead
from models.backbone import build_faster_rcnn_based_backbone


class BaseNet(GeneralizedRCNN):
    def __init__(self, backbone, rpn, roi_head, transform):
        super(BaseNet, self).__init__(backbone, rpn, roi_head, transform)

    def forward(self, images, targets):
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
        rpn_features = {"feat_res4": features_dict["feat_res4"]}
        proposals, proposal_losses = self.rpn(images, rpn_features, targets)
        detections, detector_losses = self.roi_heads(
            features_dict, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return detections, losses

    def preprocess(self, images, targets):
        images, targets = self.transform(images, targets)
        return images, targets

    @staticmethod
    def rescale_boxes(boxes, sizes, target_sizes):
        """ Rescale_boxes in batch.
        args:
            - boxes: List[Tensor], in x1y1x2y2 format.
            - sizes: List[tuple(h, w)]
            - target_sizes: List[tuple(h, w)]
        """
        device = boxes[0].device
        ratios = [
            (
                torch.as_tensor(ts[0] / s[0]).type(torch.float32).to(device),
                torch.as_tensor(ts[1] / s[1]).type(torch.float32).to(device)
            )
            for s, ts in zip(sizes, target_sizes)
        ]
        scaled_boxes = []
        for i in range(len(boxes)):
            rois = boxes[i]
            hr, wr = ratios[i]
            ratios = torch.stack([wr, hr, wr, hr]).view(1, 4)
            rois = rois * ratios
            scaled_boxes.append(rois)
        return scaled_boxes

    @torch.no_grad()
    def extract_features_with_boxes(self, images, targets, feature_norm=True):
        """ extract features with boxes.
        """
        images, targets = self.transform(images, targets)
        features_dict = self.backbone(images.tensors)

        proposals = [target["boxes"] for target in targets]
        box_features = self.roi_heads.box_roi_pool(
            features_dict, proposals, images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        embeddings, norms = self.roi_heads.reid_embed_head(box_features)

        if not feature_norm:
            embeddings = [
                embedding * norm
                for embedding, norm in zip(embeddings, norms)
            ]
        return embeddings

    @torch.no_grad()
    def extract_features_with_gtboxes(self, images, targets, feature_norm=True):
        """ extract features with boxes.
        """
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # first get all detections.
        images, _ = self.transform(images, None)
        features_dict = self.backbone(images.tensors)
        rpn_features = {"feat_res4": features_dict["feat_res4"]}
        proposals, _ = self.rpn(images, rpn_features, None)
        detections, _ = self.roi_heads(
            features_dict, proposals, images.image_sizes, targets)


        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)
        return detections

    @torch.no_grad()
    def extract_features_without_boxes(self, images):
        """ extract features with boxes.
        """
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        targets = None
        images, targets = self.transform(images, targets)
        features_dict = self.backbone(images.tensors)
        rpn_features = {"feat_res4": features_dict["feat_res4"]}
        proposals, _ = self.rpn(images, rpn_features, targets)
        detections, _ = self.roi_heads(
            features_dict, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)
        return detections

    @torch.no_grad()
    def extract_features_by_crop(self, images, targets, feature_norm=True):
        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)),
                             targets[0]['boxes'][0].tolist())
        input_tensor = images.tensors[:, :, y1:y2 + 1, x1:x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.box_head(features)
        embeddings, norms = self.roi_heads.embedding_head(rcnn_features)

        if not feature_norm:
            embeddings = [
                embedding * norm
                for embedding, norm in zip(embeddings, norms)
            ]
        return embeddings


class PSRoIHead(RoIHeads):
    """ Additional Branch for person search output.
    """
    def __init__(
            self,
            box_roi_pool,
            box_head,
            box_predictor,
            reid_embed_head,
            oim_loss,
            reid_feature_dim=256,
            # other Faster-RCNN parameters
            *args, **kwargs):
        super(PSRoIHead, self).__init__(
            box_roi_pool, box_head, box_predictor,
            *args, **kwargs)
        self.oim_loss = oim_loss
        self.reid_feature_dim = reid_feature_dim
        self.reid_embed_head = reid_embed_head

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if self.training:
            proposals, _, labels, regression_targets, pid_labels\
                = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        embeddings, norms = self.reid_embed_head(box_features)
        pred_boxes_features = box_features["feat_res5"]
        class_logits, box_regression = self.box_predictor(pred_boxes_features)

        result = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            loss_oim = self.oim_loss(embeddings, pid_labels)
            boxes = self.box_coder.decode(box_regression, proposals)
            # postprocess training outputs
            num_persons_per_images = [len(proposal) for proposal in proposals]
            embedding_list = embeddings.split(num_persons_per_images, 0)
            box_list = boxes.split(num_persons_per_images, 0)
            norm_list = norms.split(num_persons_per_images, 0)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_oim": loss_oim,
            }
            num_images = len(num_persons_per_images)
            for idx in range(num_images):
                result.append({
                    "pid_labels": pid_labels[idx],
                    "labels": labels[idx],
                    "embeddings": embedding_list[idx],
                    "boxes": box_list[idx],
                    "norm": norm_list[idx]
                })
        else:
            boxes, scores, labels, embeddings, norms = self.postprocess_detections(
                class_logits, box_regression, proposals,
                embeddings, norms, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "embeddings": embeddings[i],
                        "norm": norms[i],
                    }
                )

        return result, losses

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_pid_labels = [t["pid_labels"] for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # mind the size of labels and pid_labels is the same with proposals.
        matched_idxs, labels, pid_labels = \
            self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_pid_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            pid_labels[img_id] = pid_labels[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, pid_labels

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_pid_labels):
        matched_idxs = []
        labels = []
        pid_labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_pid_labels_in_image\
             in zip(proposals, gt_boxes, gt_labels, gt_pid_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                pid_labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                pid_labels_in_image = gt_pid_labels_in_image[clamped_matched_idxs_in_image]
                pid_labels_in_image = pid_labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                pid_labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                pid_labels_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            pid_labels.append(pid_labels_in_image)
        return matched_idxs, labels, pid_labels

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               embeddings,      # type: Tensor
                               norms,           # type: Tensor
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_embedding_list = embeddings.split(boxes_per_image, 0)
        pred_norm_list = norms.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        all_norms = []
        # TODO: embeddings and norms
        for boxes, scores, embedding, norm, image_shape in \
            zip(pred_boxes_list, pred_scores_list, pred_embedding_list, pred_norm_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            embedding, norm = embedding[inds], norm[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            embedding, norm = embedding[keep], norm[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            embedding, norm = embedding[keep], norm[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embedding)
            all_norms.append(norm)

        return all_boxes, all_scores, all_labels, all_embeddings, all_norms


def build_faster_rcnn_based_models(args):

    min_size = 800
    max_size = 1333

    # build tranform
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    # RPN parameters
    rpn_pre_nms_top_n_train = args.model.rpn.pre_nms_top_n_train
    rpn_pre_nms_top_n_test = args.model.rpn.pre_nms_top_n_test
    rpn_post_nms_top_n_test = args.model.rpn.post_nms_topn_test
    rpn_post_nms_top_n_train = args.model.rpn.post_nms_topn_train

    rpn_nms_thresh = args.model.rpn.nms_thresh
    rpn_fg_iou_thresh = args.model.rpn.pos_thresh_train
    rpn_bg_iou_thresh = args.model.rpn.neg_thresh_train

    rpn_batch_size_per_image = args.model.rpn.batch_size_train
    rpn_positive_fraction = args.model.rpn.pos_frac_train

    # Box parameters
    box_fg_iou_thresh = args.model.roi_head.pos_thresh_train
    box_bg_iou_thresh = args.model.roi_head.neg_thresh_train
    box_batch_size_per_image = args.model.roi_head.batch_size_train
    box_positive_fraction = args.model.roi_head.pos_frac_train
    bbox_reg_weights = None

    box_score_thresh = args.model.roi_head.score_thresh_test
    box_nms_thresh = args.model.roi_head.nms_thresh_test
    box_detections_per_img = args.model.roi_head.detections_per_image_test

    # model parameters
    use_multi_scale = args.model.use_multi_scale
    reid_feature_dim = args.model.reid_feature_dim

    # build backbone
    backbone, box_head = build_faster_rcnn_based_backbone(
            args.model.backbone.name,
            args.model.backbone.pretrained,
            args.model.backbone.norm_layer,
            return_res4=use_multi_scale)

    out_channels = backbone.out_channels
    # build rpn
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    rpn_head = RPNHead(
        out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
    )
    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
    rpn = RegionProposalNetwork(
        rpn_anchor_generator, rpn_head,
        rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        rpn_batch_size_per_image, rpn_positive_fraction,
        rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

    # build roi_heads
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['feat_res3', 'feat_res4'],
        output_size=14,
        sampling_ratio=2)

    representation_size = 2048  # for feat_res4 outputs
    num_classes = 2
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    # build oim
    num_features = reid_feature_dim
    num_pids = args.loss.oim.num_pids
    num_cq_size = args.loss.oim.num_cq_size
    oim_momentum = args.loss.oim.oim_momentum
    oim_scalar = args.loss.oim.oim_scalar
    oim_loss = OIMLoss(num_features, num_pids, num_cq_size, oim_momentum, oim_scalar)

    # build reid head
    if use_multi_scale:
        reid_head = ReIDEmbeddingHead(
            featmap_names=["feat_res4", "feat_res5"],
            in_channels=[1024, 2048],
            dim=reid_feature_dim, feature_norm=True)
    else:
        reid_head = ReIDEmbeddingHead(
            featmap_names=['feat_res5'], in_channels=[2048],
            dim=reid_feature_dim, feature_norm=True)

    roi_head = PSRoIHead(
        box_roi_pool, box_head, box_predictor,
        reid_head, oim_loss,
        reid_feature_dim=256,
        # RoIHead training parameters
        fg_iou_thresh=box_fg_iou_thresh, bg_iou_thresh=box_bg_iou_thresh,
        batch_size_per_image=box_batch_size_per_image,
        positive_fraction=box_positive_fraction,
        bbox_reg_weights=bbox_reg_weights,
        # RoIHead inference parameters
        score_thresh=box_score_thresh,
        nms_thresh=box_nms_thresh,
        detections_per_img=box_detections_per_img,
    )

    model = BaseNet(backbone, rpn, roi_head, transform)
    return model


if __name__ == '__main__':
    from datasets.cuhk_sysu import CUHK_SYSU
    import torchvision.transforms as T
    from datasets.transforms import ToTensor
    from configs.faster_rcnn_default_configs import get_default_cfg
    from easydict import EasyDict
    from utils import ship_to_cuda
    from datasets import build_train_cuhk
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import RandomSampler, BatchSampler

    args = get_default_cfg()

    root = "data/cuhk-sysu"
    transforms = ToTensor()

    # dataset = CUHK_SYSU(root, transforms, "train")
    dataset = build_train_cuhk(root)

    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, collate_fn=lambda x: x)

    from IPython import embed
    embed()

    image1, target1 = dataset[0]
    image2, target2 = dataset[2]

    # image_mean = [0.485, 0.456, 0.406]
    # image_std = [0.229, 0.224, 0.225]
    # min_size = 1500
    # max_size = 900
    # transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    # images, targets = transform.forward([image1, image2], [target1, target2])

    device = "cuda"
    # device = "cpu"
    device = torch.device(device)
    images = [image1, image2]
    targets = [target1, target2]
    images, targets = ship_to_cuda(images, targets, device)

    # draw boxes
    # from evaluation.utils import draw_boxes_text
    # for i in range(len(images.tensors)):
    #     img = images.tensors[i]
    #     boxes = targets[i]["boxes"]
    #     draw_boxes_text(img, boxes)

    model = build_faster_rcnn_based_models(args)
    model.to(device)

    # model.eval()

    # with torch.no_grad():
    #     outputs = model(images, targets)

    outputs = model(images, targets)

    from IPython import embed
    embed()
