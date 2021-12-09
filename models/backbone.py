from collections import defaultdict
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import FeaturePyramidNetwork, \
    BackboneWithFPN
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.resnet import __all__


class FrozenBatchNorm1d(nn.Module):
    """
    BatchNorm1d where the batch statistics and the affine parameters are fixed
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm1d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm1d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        # if x.dtype == torch.float16:
        #     self.weight = self.weight.half()
        #     self.bias = self.bias.half()
        #     self.running_mean = self.running_mean.half()
        #     self.running_var = self.running_var.half()

        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)
        return x * scale + bias


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def get_norm_layer(norm_layer):
    defined_norm_layers = defaultdict(lambda: nn.BatchNorm2d)
    defined_norm_layers.update({
        "bn": nn.BatchNorm2d,
        "frozen_bn": FrozenBatchNorm2d
    })
    return defined_norm_layers[norm_layer]


def build_backbone(
        backbone_name="resnet50", pretrained=True,
        return_layers=None, norm_layer="bn"):
    assert backbone_name in __all__, f"{backbone_name} not found."
    norm_layer = get_norm_layer(norm_layer)
    backbone = getattr(torchvision.models, backbone_name)(
        pretrained=pretrained,
        norm_layer=norm_layer
    )

    # freeeze the first layers
    for name, param in backbone.named_parameters():
        if "layer1" in name or "layer2" in name or \
                "layer3" in name or "layer4" in name:
            continue
        param.requires_grad_(False)

    if return_layers is None:
        return_layers = {"layer4": "feat0"}
    backbone = IntermediateLayerGetter(backbone, return_layers)
    return backbone


class Backbone(nn.Sequential):
    def __init__(self, backbone):
        super(Backbone, self).__init__(
            OrderedDict([
                ['conv1', backbone.conv1],
                ['bn1', backbone.bn1],
                ['relu', backbone.relu],
                ['maxpool', backbone.maxpool],
                ['layer1', backbone.layer1],  # res2
                ['layer2', backbone.layer2],  # res3
                ['layer3', backbone.layer3]]  # res4
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([['feat_res4', feat]])


class RCNNConvHead(nn.Sequential):
    def __init__(self, backbone, return_res4=True, GAP=True):
        super(RCNNConvHead, self).__init__(
            OrderedDict(
                [['layer4', backbone.layer4]]  # res5
            )
        )
        self.return_res4 = return_res4
        if self.return_res4:
            self.out_channels = [1024, 2048]
        else:
            self.out_channels = [2048]
        self.GAP = GAP

    def forward(self, x):
        feat = super(RCNNConvHead, self).forward(x)
        if self.GAP:
            x = F.adaptive_max_pool2d(x, 1)
            feat = F.adaptive_max_pool2d(feat, 1)
        if self.return_res4:
            return OrderedDict([
                ['feat_res4', x],  # Global average pooling
                ['feat_res5', feat]]
            )
        else:
            return OrderedDict([['feat_res5', feat]])


class MSBoxHead(nn.Module):
    def __init__(self, in_channels, representation_size,
                 return_res4=True, GAP=True, *args, **kwargs):
        super(MSBoxHead, self).__init__()
        self.return_res4 = return_res4
        if self.return_res4:
            self.out_channels = [1024, 2048]
        else:
            self.out_channels = [2048]
        self.GAP = GAP

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, representation_size, 3, 1, 1),
            nn.BatchNorm2d(representation_size),
            nn.ReLU(True),
            nn.Conv2d(representation_size, representation_size, 3, 1, 1),
            nn.BatchNorm2d(representation_size),
            nn.ReLU(True),
            nn.Conv2d(representation_size, self.out_channels[-1], 3, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        if self.GAP:
            x = F.adaptive_max_pool2d(x, 1)
            feat = F.adaptive_max_pool2d(feat, 1)
        if self.return_res4:
            return OrderedDict([
                ['feat_res4', x],  # Global average pooling
                ['feat_res5', feat]]
            )
        else:
            return OrderedDict([['feat_res5', feat]])


class MSTwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size, resolution,
                 return_res4=True, GAP=True, *args, **kwargs):
        super(MSTwoMLPHead, self).__init__()
        self.return_res4 = return_res4
        if self.return_res4:
            self.out_channels = [in_channels, representation_size]
        else:
            self.out_channels = [representation_size]
        self.GAP = GAP
        self.fc6 = nn.Linear(in_channels * resolution ** 2, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        feat = x.flatten(start_dim=1)
        feat = F.relu(self.fc6(feat))
        feat = F.relu(self.fc7(feat))

        if self.GAP:
            x = F.adaptive_max_pool2d(x, 1).flatten(start_dim=1)
        if self.return_res4:
            return OrderedDict([
                ['feat_res4', x],  # Global average pooling
                ['feat_res5', feat]]
            )
        else:
            return OrderedDict([['feat_res5', feat]])


def build_faster_rcnn_based_backbone(
        backbone_name, pretrained, norm_layer="bn",
        return_res4=False, GAP=True):
    assert backbone_name in __all__, f"{backbone_name} not found."
    norm_layer = get_norm_layer(norm_layer)
    backbone = getattr(torchvision.models, backbone_name)(
        pretrained=pretrained,
        norm_layer=norm_layer
    )

    # freeze layers
    backbone.conv1.weight.requires_grad_(False)
    backbone.bn1.weight.requires_grad_(False)
    backbone.bn1.bias.requires_grad_(False)

    stem = Backbone(backbone)
    head = RCNNConvHead(backbone, return_res4, GAP)
    return stem, head


def build_faster_rcnn_based_multi_scale_backbone(
        backbone_name, pretrained,
        use_fpn=False,
        norm_layer="bn",
        return_res4=False, GAP=True):

    assert backbone_name in __all__, f"{backbone_name} not found."
    # norm_layer = "frozen_bn"
    norm_layer = get_norm_layer(norm_layer)
    backbone = getattr(torchvision.models, backbone_name)(
        pretrained=pretrained,
        norm_layer=norm_layer
    )

    # freeze layers
    backbone.conv1.weight.requires_grad_(False)
    backbone.bn1.weight.requires_grad_(False)
    backbone.bn1.bias.requires_grad_(False)

    # # freeze the layer1
    # for name, param in backbone.named_parameters():
    #     if name.startswith("layer1"):
    #         param.requires_grad_(False)

    # split_stem = Backbone(backbone)
    # head = RCNNConvHead(backbone, return_res4, GAP)
    stem = BackboneWithFPN(
        backbone,
        return_layers=dict(
            layer1="feat_res2",
            layer2="feat_res3",
            layer3="feat_res4",
            layer4="feat_res5"
        ),
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
    )
    # HACK: out_channel is fixed
    setattr(stem, "out_channels", 256)
    # head = MSBoxHead(
    #     stem.out_channels, 1024, return_res4, GAP)
    head = MSTwoMLPHead(
        stem.out_channels, 1024, 7,
        return_res4, GAP
    )
    return stem, head


def build_fpn_backbone(
        backbone_name, pretrained, norm_layer="bn"):

    assert backbone_name in __all__, f"{backbone_name} not found."
    norm_layer = get_norm_layer(norm_layer)
    backbone = getattr(torchvision.models, backbone_name)(
        pretrained=pretrained,
        norm_layer=norm_layer
    )

    # freeze layers
    backbone.conv1.weight.requires_grad_(False)
    backbone.bn1.weight.requires_grad_(False)
    backbone.bn1.bias.requires_grad_(False)

    stem = BackboneWithFPN(
        backbone,
        return_layers=dict(
            layer1="feat_res2",
            layer2="feat_res3",
            layer3="feat_res4",
            layer4="feat_res5"
        ),
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
    )
    return stem
