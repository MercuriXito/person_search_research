from yacs.config import CfgNode as CN
import configs.faster_rcnn_default_configs as fconfig

# default configuration for graph_net models
_C = fconfig.get_default_cfg()

# -------------------------------------------------------- #
#             Additional Contex Graph Head                 #
# -------------------------------------------------------- #
_C.model.graph_head.use_graph = True
_C.model.graph_head.num_graph_stack = 1
_C.model.graph_head.nheads = 4
_C.model.graph_head.dropout = 0.0

# -------------------------------------------------------- #
#                        Train                             #
# -------------------------------------------------------- #
# dataset, dataloader
_C.train.image_set = "train"
_C.train.dataset = "cuhk-sysu"
_C.train.data_root = "data/cuhk-sysu"
_C.train.aug = "multi_scale_with_crop"

_C.train.batch_size = 4
_C.train.num_workers = 4

_C.train.epochs = 20
_C.train.clip_max_norm = 0.1
_C.train.resume = ""
_C.train.output_dir = "exps/test.graph"

_C.train.lr = 1e-4
_C.train.weight_decay = 1e-4
_C.train.lr_drop_epochs = [10]

_C.train.loss_weights.loss_classifier = 2.0
_C.train.loss_weights.loss_box_reg = 5.0
_C.train.loss_weights.loss_oim = 1.0
_C.train.loss_weights.loss_objectness = 2.0
_C.train.loss_weights.loss_rpn_box_reg = 5.0
_C.train.loss_weights.loss_graph = 1.0

_C.train.graph.use_ema_lut = False
_C.train.graph.lut_momentum = 0.1

# -------------------------------------------------------- #
#                           eval                           #
# -------------------------------------------------------- #

_C.eval.det_thresh = 0.01
_C.eval.gallery_size = 100
_C.eval.graph_thred = 0.0
_C.eval.nms_thresh = 0.5
_C.eval.use_data = ""
_C.eval.device = "cuda"
_C.eval.dataset_file = "cuhk-sysu"
_C.eval.dataset_path = "data/cuhk-sysu"
_C.eval.eval_context = False
_C.eval.checkpoint = "checkpoint.pth"


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()


def equal(fargs, gargs, ignore_value=True, prefix=""):
    """ iteratively check if two args are equal.
    """
    assert isinstance(fargs, CN)
    assert isinstance(gargs, CN)

    fkeys = set(fargs.keys())
    gkeys = set(gargs.keys())

    inter = fkeys.intersection(gkeys)
    outer = fkeys.union(gkeys) - inter
    if len(outer) != 0:
        print("Not equal key:")
        print([f"{prefix}.{ckey}" for ckey in list(outer)])

    for key in inter:
        fitem = fargs[key]
        gitem = gargs[key]
        if len(prefix) > 0:
            full_key_name = ".".join([prefix, key])
        else:
            full_key_name = key
        if not ignore_value and fitem != gitem:
            print(f"Value of {full_key_name} not equal")
        if isinstance(fitem, CN) and isinstance(gitem, CN):
            equal(fitem, gitem, ignore_value, full_key_name)


def test():
    gargs = get_default_cfg()
    fargs = fconfig.get_default_cfg()
    equal(fargs, gargs, False)
