from .baseline import build_faster_rcnn_based_models
# other ps models
from .baseline_fpn import build_faster_rcnn_based_models as build_fpn_models
from .baseline_retinanet import build_retinanet_based_models
from .baseline_anchor_free import build_anchor_free_based_models, \
    build_anchor_free_base_models_with_reid_head
# graph models
from .graph_net import build_graph_net
from .baseline_retinanet_graph import build_retinanet_graph
from .baseline_anchor_free_graph import build_anchor_free_graph
from .baseline_fpn_graph import build_fpn_graph


def build_models(args):
    # compatible with previous args definition
    if not hasattr(args.model, "name"):
        model_name = "fasterrcnn"
    else:
        model_name = args.model.name
    print(model_name)
    if model_name == "fasterrcnn":
        return build_faster_rcnn_based_models(args)
    elif model_name == "fpn":
        return build_fpn_models(args)
    elif model_name == "retinanet":
        return build_retinanet_based_models(args)
    elif model_name == "fcos":
        # return build_anchor_free_based_models(args)
        return build_anchor_free_base_models_with_reid_head(args)
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
    elif model_name == "fpn":
        return build_fpn_graph(args)
    elif model_name == "retinanet":
        return build_retinanet_graph(args)
    elif model_name == "fcos":
        return build_anchor_free_graph(args)
    else:
        raise NotImplementedError(f"{model_name}")
