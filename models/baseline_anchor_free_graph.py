import torch

from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision.ops.boxes as box_ops
from torchvision.ops.poolers import MultiScaleRoIAlign

from models.losses import OIMLoss
from models.backbone import build_fpn_backbone, \
    build_faster_rcnn_based_multi_scale_backbone
from models.baseline_retinanet import PSRoIHead
from models.reid_head import ReIDEmbeddingHead
from models.baseline_anchor_free import AnchorFreePS, FCOSHead
from models.ctx_attn_head import build_criterion_for_graph_head, build_graph_head


class GraphAnchorFreePS(AnchorFreePS):
    def __init__(
            self,
            backbone,
            fcos_head,
            transform,
            # added graph head
            graph_head=None,
            # ps roi head
            ps_roi_head=None,
            # other parameters
            topk_candidates=1000,
            detections_per_img=300,
            score_thresh=0.05,
            nms_thresh=0.5,
            ):

        super().__init__(
            backbone, fcos_head, transform, ps_roi_head, topk_candidates,
            detections_per_img, score_thresh, nms_thresh,
        )
        self.graph_head = graph_head

    def forward(self, images, targets=None, feats_lut=None):
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

        detections, loss = {}, {}
        if self.training:
            loss, matched_idxs = \
                self.compute_loss(points, head_outputs, targets, images.image_sizes, feats_shapes)
            detections = head_outputs

            if self.ps_roi_head is not None:
                pro_regression = head_outputs["bbox_regression"].unbind(dim=0)
                pro_points = points.unbind(dim=0)
                proposals = self.box_coder.decode(
                    pro_regression, pro_points, images.image_sizes)
                # assert isinstance(proposals, torch.Tensor)
                proposals = proposals.split(pro_points[0].shape[0])
                assert isinstance(self.ps_roi_head, PSRoIHead)
                ps_outs, ps_loss = self.ps_roi_head.forward(
                    features_dict, proposals, images.image_sizes,
                    targets, matched_idxs=matched_idxs, return_sampled_inds=True)
                loss.update(ps_loss)

            if self.graph_head is not None:
                graph_outs, loss_graph = self.training_graph_forward(
                    proposals, head_outputs, ps_outs, targets, feats_lut)
                loss.update(loss_graph)
        else:
            detections = self.postprocess(points, head_outputs, feats_shapes, images.image_sizes)
            if self.ps_roi_head is not None:
                # append outputs from roi head
                proposals = [det["boxes"] for det in detections]
                # TODO: 3. features_levels in inference
                ps_outs, _ = self.ps_roi_head.forward(features_dict, proposals, images.image_sizes)
                for i in range(len(detections)):
                    detections[i].update(ps_outs[i])

            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections, loss

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
            # 1. batched_nms
            keep = box_ops.batched_nms(sample_boxes,
                                       sample_scores,
                                       pid_label_in_image,
                                       iou_threshold=0.5)
            for k, v in det_item.items():
                det_item[k] = v[keep]
            detections.append(det_item)

        graph_outs, loss_graph = self.graph_head.forward(detections, targets, feats_lut)
        return graph_outs, loss_graph


def build_anchor_free_graph(args):
    min_size = 800
    max_size = 1333

    # build tranform
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    # build fpn backbone
    backbone, roi_net = build_fpn_backbone(
            args.model.backbone.name,
            args.model.backbone.pretrained,
            args.model.backbone.norm_layer,
            build_head=True,
            rep_size=args.model.roi_head.layer4_rep_size)
    num_classes = 1  # excluding the background
    fcos_head = FCOSHead(256, num_classes=num_classes)

    # model parameters
    use_multi_scale = args.model.use_multi_scale
    reid_feature_dim = args.model.reid_feature_dim
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

    model = GraphAnchorFreePS(
        backbone,
        fcos_head,
        transform,
        graph_head=graph_head,
        ps_roi_head=ps_roi_head)
    return model


if __name__ == '__main__':
    from datasets.transforms import ToTensor
    from configs.faster_rcnn_default_configs import get_default_cfg
    from utils.misc import ship_to_cuda
    from datasets import build_trainset

    args = get_default_cfg()
    device = "cuda"
    device = torch.device(device)

    root = "data/cuhk-sysu"
    transforms = ToTensor()
    dataset = build_trainset("cuhk-sysu", root, False)
    image1, target1 = dataset[0]
    image2, target2 = dataset[1]
    images = [image1, image2]
    targets = [target1, target2]
    images, targets = ship_to_cuda(images, targets, device)

    model = build_anchor_free_graph(args)
    model.to(device)
    outputs = model(images, targets)

    from IPython import embed
    embed()
