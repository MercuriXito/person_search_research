import torch
from collections import OrderedDict

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.poolers import MultiScaleRoIAlign
import torchvision.ops.boxes as box_ops

from models.baseline_retinanet import RetinaNet
from models.losses import OIMLoss
from models.backbone import build_fpn_backbone, \
    build_faster_rcnn_based_multi_scale_backbone
from models.reid_head import ReIDEmbeddingHead
from models.baseline_retinanet import PSRoIHead, RetinaNetHead
from models.ctx_attn_head import build_criterion_for_graph_head, build_graph_head


class GraphRetinaNet(RetinaNet):
    def __init__(self, backbone, num_classes,
                 transform,
                 # ps roi head
                 ps_roi_head=None,
                 # graph head
                 graph_head=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000):
        super().__init__(
            backbone, num_classes, transform, ps_roi_head, anchor_generator, head,
            # default parameters
            proposal_matcher, score_thresh, nms_thresh,
            detections_per_img, fg_iou_thresh, bg_iou_thresh,
            topk_candidates
        )
        self.graph_head = graph_head

    def num_anchors_per_level(self, features_list, full_size: int):
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features_list]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = full_size
        A = HWA // HW  # number of anchors for one positiion.
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]
        return num_anchors_per_level

    def forward(self, images, targets=None, feats_lut=None):
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
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)
        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        features_list = list(features.values())
        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features_list)
        # create the set of anchors
        anchors = self.anchor_generator(images, features_list)

        losses = {}
        detections = []

        if self.training:
            assert targets is not None
            # losses of baseline.
            detection_losses, matched_idxs = self.compute_loss(targets, head_outputs, anchors)
            proposals = self.box_coder.decode(head_outputs["bbox_regression"], anchors)
            proposals = proposals.view(len(anchors), -1, 4).unbind(dim=0)
            ps_outs, ps_loss = self.ps_roi_head.forward(
                features, proposals, images.image_sizes,
                targets, matched_idxs=matched_idxs, images=images, return_sampled_inds=True)
            losses.update(detection_losses)
            losses.update(ps_loss)

            graph_outs, loss_graph = self.training_graph_forward(
                proposals, head_outputs, ps_outs, targets, feats_lut)
            losses.update(loss_graph)
        else:
            num_anchors_per_level = self.num_anchors_per_level(
                features_list, head_outputs['cls_logits'].size(1))
            # split outputs per level
            split_head_outputs = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            # append outputs from roi head
            proposals = [det["boxes"] for det in detections]
            ps_outs, _ = self.ps_roi_head.forward(features, proposals, images.image_sizes)
            for i in range(len(detections)):
                detections[i].update(ps_outs[i])
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        return detections, losses

    def training_graph_forward(self,
                               proposals,
                               head_outputs,
                               ps_outs,
                               targets=None,
                               feats_lut=None):
        """ Pack input for graph module.
        """
        # Grab Graph Input.
        scores = torch.sigmoid(head_outputs["cls_logits"]).flatten(start_dim=1)
        scores = scores.unbind(dim=0)
        ps_sampled_inds, sampled_pid_labels, sampled_embeddings = [
                [out[name] for out in ps_outs]
                for name in ["sampled_inds", "pid_labels", "embeddings"]
            ]

        # pre-process
        detections = []
        for boxes_in_img, scores_in_imgs, pid_label_in_image, embedding_in_img, sampled_ind in \
                    zip(proposals, scores, sampled_pid_labels, sampled_embeddings, ps_sampled_inds):
            sample_boxes = boxes_in_img[sampled_ind]
            sample_scores = scores_in_imgs[sampled_ind]

            det_item = dict(boxes=sample_boxes,
                            scores=sample_boxes,
                            pid_labels=pid_label_in_image,
                            embeddings=embedding_in_img)
            keep = box_ops.batched_nms(sample_boxes,
                                       sample_scores,
                                       pid_label_in_image,
                                       iou_threshold=0.5)
            for k, v in det_item.items():
                det_item[k] = v[keep]
            detections.append(det_item)

        graph_outs, loss_graph = self.graph_head.forward(detections, targets, feats_lut)
        return graph_outs, loss_graph


def build_retinanet_graph(args):
    # build backbone
    backbone, roi_net = build_fpn_backbone(
            args.model.backbone.name,
            args.model.backbone.pretrained,
            args.model.backbone.norm_layer,
            build_head=True,
            rep_size=args.model.roi_head.layer4_rep_size)
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
    if args.model.roi_head.use_layer4:
        box_head = roi_net
    else:
        _, box_head = build_faster_rcnn_based_multi_scale_backbone(
                args.model.backbone.name,
                args.model.backbone.pretrained,
                args.model.backbone.norm_layer,
                return_res4=use_multi_scale)

    # build reid head
    representation_size = box_head.out_channels
    if use_multi_scale:
        reid_head = ReIDEmbeddingHead(
            featmap_names=["feat_res4", "feat_res5"],
            in_channels=representation_size,
            dim=reid_feature_dim, feature_norm=True)
    else:
        reid_head = ReIDEmbeddingHead(
            featmap_names=['feat_res5'],
            in_channels=[representation_size[-1]],
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

    # graph module
    graph_loss = build_criterion_for_graph_head(args.model.graph_head.loss)
    graph_stack = args.model.graph_head.num_graph_stack
    graph_nheads = args.model.graph_head.nheads
    graph_dropout = args.model.graph_head.dropout
    graph_module = args.model.graph_head.graph_module
    graph_head = build_graph_head(
        module=graph_module,
        criterion=graph_loss,
        num_pids=num_pids,
        reid_feature_dim=256,
        num_stack=graph_stack,
        nheads=graph_nheads,
        dropout=graph_dropout
    )

    network = GraphRetinaNet(
        backbone, num_classes,
        transform=transform,
        ps_roi_head=ps_roi_head,
        graph_head=graph_head,
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


if __name__ == '__main__':

    from configs.faster_rcnn_default_configs import get_default_cfg
    from utils.misc import ship_to_cuda
    from datasets import build_trainset
    from models.ctx_attn_head import ImageFeaturesLut

    args = get_default_cfg()

    device = "cuda"
    device = torch.device(device)

    root = "data/cuhk-sysu"
    dataset = build_trainset("cuhk-sysu", root, use_transform=False)
    lut = ImageFeaturesLut(dataset)

    model = build_retinanet_graph(args)
    # checkpoint_path = "exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048/checkpoint.pth"
    # params = torch.load(checkpoint_path, map_location="cpu")
    # model_params = params["model"]
    # missed, unexpected = model.load_state_dict(model_params, strict=False)

    model.to(device)

    images, targets = [], []
    for i in range(2):
        image, target = dataset[i]
        images.append(image)
        targets.append(target)

    images, targets = ship_to_cuda(images, targets, device)
    detections, losses = model(images, targets, lut)
    from IPython import embed
    embed()
