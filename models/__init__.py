from .baseline import build_faster_rcnn_based_models
from .baseline_fpn import build_faster_rcnn_based_models as build_fpn_models
from .baseline_retinanet import build_retinanet_based_models
from .baseline_anchor_free import build_anchor_free_based_models

from .graph_net import build_graph_net


def build_models(args):
    # compatible with previous args definition
    if not hasattr(args.model, "name"):
        model_name = "fasterrcnn"
    else:
        model_name = args.model.name
    if model_name == "fasterrcnn":
        return build_faster_rcnn_based_models(args)
    elif model_name == "fpn":
        return build_fpn_models(args)
    elif model_name == "retinanet":
        return build_retinanet_based_models(args)
    elif model_name == "fcos":
        return build_anchor_free_based_models(args)
    else:
        return build_faster_rcnn_based_models(args)


def build_graph_models(args):
    # compatible with previous args definition
    if not hasattr(args.model, "name"):
        model_name = "fasterrcnn"
    else:
        model_name = args.model.name
    if model_name == "fasterrcnn":
        return build_graph_net(args)
    else:
        raise NotImplementedError(f"{model_name}")
