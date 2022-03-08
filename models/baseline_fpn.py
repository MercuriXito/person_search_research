import torch
import torch.nn.functional as F

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from models.losses import OIMLoss
from models.reid_head import ReIDEmbeddingHead
from models.backbone import build_faster_rcnn_based_multi_scale_backbone
from models.baseline import PSRoIHead


class BaseFPNNet(GeneralizedRCNN):
    def __init__(self, backbone, rpn, roi_head, transform):
        super(BaseFPNNet, self).__init__(backbone, rpn, roi_head, transform)

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
        # rpn_features = {"feat_res4": features_dict["feat_res4"]}
        # rpn_features = dict([(name, val) for name, val in features_dict.items() if "feat" in name])
        # proposals, proposal_losses = self.rpn(images, rpn_features, targets)
        proposals, proposal_losses = self.rpn(images, features_dict, targets)
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
        # rpn_features = {"feat_res4": features_dict["feat_res4"]}
        rpn_features = dict([(name, val) for name, val in features_dict.items() if "feat" in name])
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
        # rpn_features = {"feat_res4": features_dict["feat_res4"]}
        rpn_features = dict([(name, val) for name, val in features_dict.items() if "feat" in name])
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


def build_faster_rcnn_based_models(args):

    # multi_scale 其实只影响两点: (1) box_head 的输出； (2) reid_head 的 feature 计算
    # TODO: add multis_scale support for FPN based models.
    # if hasattr(args.model, "use_multi_scale") and args.model.use_multi_scale:
    #     warnings.warn("Multi Scale not for FPN based model, set to False.")
    #     args.defrost()
    #     args.model.use_multi_scale = False
    #     args.freeze()

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
    roi_use_ksampling = args.model.roi_head.k_sampling
    roi_use_k = args.model.roi_head.k

    # model parameters
    use_multi_scale = args.model.use_multi_scale
    reid_feature_dim = args.model.reid_feature_dim

    # build backbone
    # backbone, box_head = build_faster_rcnn_based_backbone(
    #         args.model.backbone.name,
    #         args.model.backbone.pretrained,
    #         args.model.backbone.norm_layer,
    #         return_res4=use_multi_scale)

    backbone, box_head = build_faster_rcnn_based_multi_scale_backbone(
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
        featmap_names=['feat_res2', 'feat_res3', 'feat_res4', 'feat_res5'],
        output_size=7,
        sampling_ratio=2)

    representation_size = box_head.out_channels[-1]
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
    # reid_head = ReIDEmbeddingHead(
    #     featmap_names=['feat_res5'], in_channels=[box_head.out_channels[-1]],
    #     dim=reid_feature_dim, feature_norm=True)

    roi_head = PSRoIHead(
        box_roi_pool, box_head, box_predictor,
        reid_head, oim_loss,
        reid_feature_dim=256,
        # RoIHead training parameters
        fg_iou_thresh=box_fg_iou_thresh, bg_iou_thresh=box_bg_iou_thresh,
        batch_size_per_image=box_batch_size_per_image,
        positive_fraction=box_positive_fraction,
        bbox_reg_weights=bbox_reg_weights,
        # Sampling parameters,
        k_sampling=roi_use_ksampling,
        k=roi_use_k,
        # RoIHead inference parameters
        score_thresh=box_score_thresh,
        nms_thresh=box_nms_thresh,
        detections_per_img=box_detections_per_img,
        # GraphHead parameters
        graph_head=None
    )

    model = BaseFPNNet(backbone, rpn, roi_head, transform)
    return model


if __name__ == '__main__':
    # testing model
    from torchvision.transforms import ToTensor
    from configs.faster_rcnn_default_configs import get_default_cfg
    from datasets import build_trainset
    from torch.utils.data.sampler import RandomSampler, BatchSampler
    from torch.utils.data import DataLoader
    from utils.misc import ship_to_cuda

    # load dataset, build input
    root = "data/cuhk-sysu"
    transforms = ToTensor()
    dataset = build_trainset("cuhk-sysu", root)

    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, collate_fn=lambda x: x)
    image1, target1 = dataset[0]
    image2, target2 = dataset[1]

    device = "cuda"
    device = torch.device(device)
    images = [image1, image2]
    targets = [target1, target2]
    images, targets = ship_to_cuda(images, targets, device)

    # build
    args = get_default_cfg()
    model = build_faster_rcnn_based_models(args)
    model.to(device)

    with torch.no_grad():
        outputs = model(images, targets)

    from IPython import embed
    embed()
