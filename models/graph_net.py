import torch
import torch.nn.functional as F
from copy import deepcopy

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from models.losses import OIMLoss
from models.reid_head import ReIDEmbeddingHead
from models.backbone import build_faster_rcnn_based_backbone
from models.baseline import BaseNet, PSRoIHead
from models.ctx_attn_head import ImageFeaturesLut, build_criterion_for_graph_head, build_graph_head


class GraphNet(BaseNet):
    def __init__(self, backbone, rpn, roi_head, transform, graph_head):
        super().__init__(backbone, rpn, roi_head, transform)
        self.graph_head = graph_head

    def forward(self, images, targets, feats_lut=None):
        """
        args:
            - image: List[Tensor]
            - targets: List[Dict(str, Tensor)]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training and feats_lut is None:
            raise ValueError("GraphNet require feats_lut in training.")

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

        # additional branch for graph head
        graph_outs, graph_losses = self.graph_head(detections, targets, feats_lut)
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(graph_losses)

        return detections, losses


def build_graph_net(args):

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
    reid_head_norm_layer = args.model.reid_head.norm_layer
    if use_multi_scale:
        reid_head = ReIDEmbeddingHead(
            featmap_names=["feat_res4", "feat_res5"],
            in_channels=[1024, 2048],
            dim=reid_feature_dim, feature_norm=True,
            norm_layer=reid_head_norm_layer)
    else:
        reid_head = ReIDEmbeddingHead(
            featmap_names=['feat_res5'], in_channels=[2048],
            dim=reid_feature_dim, feature_norm=True,
            norm_layer=reid_head_norm_layer)

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
        # GraphHead parameters
        graph_head=None
    )

    # build graph head
    # graph_loss = deepcopy(oim_loss)
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

    model = GraphNet(backbone, rpn, roi_head, transform, graph_head)
    return model


if __name__ == '__main__':
    from datasets.cuhk_sysu import CUHK_SYSU
    import torchvision.transforms as T
    from datasets.transforms import ToTensor
    from configs.graph_net_default_configs import get_default_cfg
    from easydict import EasyDict
    from utils.misc import ship_to_cuda
    from datasets import build_trainset
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import RandomSampler, BatchSampler

    args = get_default_cfg()

    root = "data/cuhk-sysu"
    transforms = ToTensor()

    # dataset = CUHK_SYSU(root, transforms, "train")
    dataset = build_trainset("cuhk-sysu", root)

    lut = ImageFeaturesLut(dataset)

    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, collate_fn=lambda x: x)

    image1, target1 = dataset[0]
    image2, target2 = dataset[1]

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
    # from utils.vis import draw_boxes_text
    # for i in range(len(images.tensors)):
    #     img = images.tensors[i]
    #     boxes = targets[i]["boxes"]
    #     draw_boxes_text(img, boxes)

    model = build_graph_net(args)
    # model.load_state_dict(torch.load("exps/exps_cuhk.graph/checkpoint.pth", map_location="cpu")["model"])
    model.to(device)

    # model.eval()

    # with torch.no_grad():
    #     outputs = model(images, targets)

    outputs = model(images, targets, lut)

    from IPython import embed
    embed()
