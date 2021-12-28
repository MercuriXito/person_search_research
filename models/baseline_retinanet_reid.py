"""
RetinaNet det + reid head in fpn based model.
"""
import torch
import torch.nn as nn
from collections import OrderedDict

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from configs.faster_rcnn_default_configs import get_default_cfg

from models.losses import OIMLoss
from models.backbone import build_fpn_backbone, \
    build_faster_rcnn_based_multi_scale_backbone
from models.reid_head import ReIDEmbeddingHead
from models.baseline_retinanet import RetinaNet, RetinaNetHead
from models.baseline_fpn import PSRoIHead


def build_and_load_pretrained_fuse_model(*args, **kwargs):
    """ pretrained RetinaNet(det) + pretrained reid part of fpn baseline.
    """
    args = get_default_cfg()
    device = torch.device(args.device)
    model = build_fuse_retina_reid(args)

    # load retinanet det
    checkpoint = torch.load(
        "exps/exps_det/exps_cuhk.retinanet/checkpoint.pth",
        map_location="cpu")["model"]
    for k in list(checkpoint.keys()):
        if k.startswith("ps_roi_head"):
            checkpoint.pop(k)
    assert isinstance(checkpoint, OrderedDict)

    reid_checkpoint = torch.load(
        "exps/exps_det/exps_cuhk.fpn/checkpoint.pth",
        map_location="cpu")["model"]
    transformed_checkpoints = []
    for k, v in reid_checkpoint.items():
        if not k.startswith("roi_head"):
            continue
        key = k.replace("roi_heads", "ps_roi_head")
        transformed_checkpoints.append((key, v))
    transformed_checkpoints = OrderedDict(transformed_checkpoints)
    checkpoint.update(transformed_checkpoints)

    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model, args


class FuseRoiHead(PSRoIHead):
    """ The same Person Search ReID module as that used in
    fpn baseline, without the forward of detections.
    """
    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None,   # type: Optional[List[Dict[str, Tensor]]]
                *args, **kwargs
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

        result = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            num_persons_per_images = [len(proposal) for proposal in proposals]
            num_images = len(num_persons_per_images)
            loss_oim = self.oim_loss(embeddings, pid_labels)
            # postprocess training outputs
            embedding_list = embeddings.split(num_persons_per_images, 0)
            norm_list = norms.split(num_persons_per_images, 0)
            losses = {
                "loss_oim": loss_oim,
            }
            for idx in range(num_images):
                result.append({
                    "pid_labels": pid_labels[idx],
                    "labels": labels[idx],
                    "embeddings": embedding_list[idx],
                    "norm": norm_list[idx],
                })
        else:
            boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
            embeddings = embeddings.split(boxes_per_image, 0)
            norms = norms.split(boxes_per_image, 0)
            num_images = len(proposals)
            for i in range(num_images):
                result.append(
                    {
                        "embeddings": embeddings[i],
                        "norm": norms[i],
                    }
                )

        return result, losses


def build_fuse_retina_reid(args):
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

    # build roi_heads
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['feat_res3', 'feat_res4', 'feat_res5'],
        output_size=7,
        sampling_ratio=2)

    # representation_size = 2048  # for feat_res4 outputs
    # representation_size = 256  # for feat_res4 outputs
    representation_size = 1024  # for feat_res4 outputs
    num_classes = 2
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    use_multi_scale = args.model.use_multi_scale
    reid_feature_dim = args.model.reid_feature_dim
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
            in_channels=[256, representation_size],
            dim=reid_feature_dim, feature_norm=True)
    else:
        reid_head = ReIDEmbeddingHead(
            featmap_names=['feat_res5'], in_channels=[256],
            dim=reid_feature_dim, feature_norm=True)

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

    # build backbone
    _, box_head = build_faster_rcnn_based_multi_scale_backbone(
            args.model.backbone.name,
            args.model.backbone.pretrained,
            args.model.backbone.norm_layer,
            return_res4=use_multi_scale)

    ps_roi_head = FuseRoiHead(
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


def main():
    # testing model
    from configs.faster_rcnn_default_configs import get_default_cfg
    from datasets import build_trainset
    from utils.misc import ship_to_cuda
    from collections import OrderedDict

    # load dataset, build input
    root = "data/cuhk-sysu"
    dataset = build_trainset("cuhk-sysu", root)

    image1, target1 = dataset[0]
    image2, target2 = dataset[1]

    device = "cuda"
    device = torch.device(device)
    images = [image1, image2]
    targets = [target1, target2]
    images, targets = ship_to_cuda(images, targets, device)

    # build
    args = get_default_cfg()
    model = build_fuse_retina_reid(args)

    # load retinanet det
    checkpoint = torch.load(
        "exps/exps_det/exps_cuhk.retinanet/checkpoint.pth",
        map_location="cpu")["model"]
    for k in list(checkpoint.keys()):
        if k.startswith("ps_roi_head"):
            checkpoint.pop(k)
    assert isinstance(checkpoint, OrderedDict)

    reid_checkpoint = torch.load(
        "exps/exps_det/exps_cuhk.fpn/checkpoint.pth",
        map_location="cpu")["model"]
    transformed_checkpoints = []
    for k, v in reid_checkpoint.items():
        if not k.startswith("roi_head"):
            continue
        key = k.replace("roi_heads", "ps_roi_head")
        transformed_checkpoints.append((key, v))
    transformed_checkpoints = OrderedDict(transformed_checkpoints)
    checkpoint.update(transformed_checkpoints)

    model.load_state_dict(checkpoint, strict=True)
    # mkeys, ukeys = model.load_state_dict(transformed_checkpoints, strict=False)
    model.to(device)
    model.eval()
    model.train()

    with torch.no_grad():
        outputs, losses = model(images, targets)
    # outputs, losses = model(images, targets)

    from IPython import embed
    embed()


if __name__ == '__main__':
    main()
