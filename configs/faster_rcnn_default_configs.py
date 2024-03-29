from yacs.config import CfgNode as CN

# default configuration for faster-rcnn based models
_C = CN()
# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
# # Size of the smallest side of the image
# _C.input.min_size = 900
# # Maximum size of the side of the image
# _C.input.max_size = 1500

# # Number of images per batch
# _C.input.batch_size_train = 5
# _C.input.batch_size_test = 1

# # Number of data loading threads
# _C.input.num_workers_train = 5
# _C.input.num_workers_test = 1

_C.model = CN()
_C.model.name = "fasterrcnn"
_C.model.use_multi_scale = True
_C.model.reid_feature_dim = 256

# -------------------------------------------------------- #
#                            RPN                           #
# -------------------------------------------------------- #

_C.model.rpn = CN()
# NMS threshold used on RoIs
_C.model.rpn.nms_thresh = 0.7
# Number of anchors per image used to train RPN
_C.model.rpn.batch_size_train = 256
# Target fraction of foreground examples per RPN minibatch
_C.model.rpn.pos_frac_train = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.model.rpn.pos_thresh_train = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.model.rpn.neg_thresh_train = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.model.rpn.pre_nms_top_n_train = 12000
_C.model.rpn.pre_nms_top_n_test = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.model.rpn.post_nms_topn_train = 2000
_C.model.rpn.post_nms_topn_test = 300

# -------------------------------------------------------- #
#                         RoI head                         #
# -------------------------------------------------------- #
_C.model.roi_head = CN()
# Whether to use bn neck (i.e. batch normalization after linear)
_C.model.roi_head.bn_neck = True
# Number of RoIs per image used to train RoI head
_C.model.roi_head.batch_size_train = 128
# Target fraction of foreground examples per RoI minibatch
_C.model.roi_head.pos_frac_train = 0.5
# Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN)
_C.model.roi_head.pos_thresh_train = 0.5
# Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN)
_C.model.roi_head.neg_thresh_train = 0.5
# Minimum score threshold
_C.model.roi_head.score_thresh_test = 0.5
# NMS threshold used on boxes
_C.model.roi_head.nms_thresh_test = 0.4
# Maximum number of detected objects
_C.model.roi_head.detections_per_image_test = 300

# K-Sampling Strategies in training RoI Head.
_C.model.roi_head.k_sampling = False
_C.model.roi_head.k = 16

# subnet, only used in model with fpn
_C.model.roi_head.use_layer4 = False
_C.model.roi_head.layer4_rep_size = 1024

# -------------------------------------------------------- #
#             Additional Contex Graph Head                 #
# -------------------------------------------------------- #

_C.model.graph_head = CN()
_C.model.graph_head.use_graph = False
_C.model.graph_head.graph_module = ""
_C.model.graph_head.num_graph_stack = 1
_C.model.graph_head.nheads = 4
_C.model.graph_head.dropout = 0.0
_C.model.graph_head.loss = CN()
_C.model.graph_head.loss.name = "oim"
_C.model.graph_head.loss.num_features = 256
_C.model.graph_head.loss.num_pids = 5532
_C.model.graph_head.loss.num_cq_size = 5000
_C.model.graph_head.loss.oim_momentum = 0.5
_C.model.graph_head.loss.oim_scalar = 30.0
_C.model.graph_head.loss.margin = 0.25
_C.model.graph_head.loss.circle_gamma = 64


# -------------------------------------------------------- #
#                         backbone                         #
# -------------------------------------------------------- #
_C.model.backbone = CN()

# other parameters of models
_C.model.backbone.name = "resnet50"
_C.model.backbone.norm_layer = "bn"
_C.model.backbone.pretrained = True

# -------------------------------------------------------- #
#                reid head for person search               #
# -------------------------------------------------------- #

_C.model.reid_head = CN()
_C.model.reid_head.norm_layer = "bn"

# -------------------------------------------------------- #
#                           Loss                           #
# -------------------------------------------------------- #
_C.loss = CN()
_C.loss.oim = CN()
# Size of the lookup table in OIM
_C.loss.oim.num_pids = 5532
# Size of the circular queue in OIM
_C.loss.oim.num_cq_size = 5000
_C.loss.oim.oim_momentum = 0.5
_C.loss.oim.oim_scalar = 30.0

# -------------------------------------------------------- #
#                        Train                             #
# -------------------------------------------------------- #
_C.train = CN()

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
_C.train.output_dir = "exps/test"

_C.train.lr = 1e-4
_C.train.weight_decay = 1e-4
_C.train.lr_drop_epochs = [10]

_C.train.loss_weights = CN()
_C.train.loss_weights.loss_classifier = 2.0
_C.train.loss_weights.loss_box_reg = 5.0
_C.train.loss_weights.loss_oim = 1.0
_C.train.loss_weights.loss_objectness = 2.0
_C.train.loss_weights.loss_rpn_box_reg = 5.0
_C.train.loss_weights.loss_graph = 0.0
_C.train.loss_weights.loss_centerness = 1.0  # set for fcos

_C.train.graph = CN()
_C.train.graph.use_ema_lut = False
_C.train.graph.lut_momentum = 0.1

# -------------------------------------------------------- #
#                           eval                           #
# -------------------------------------------------------- #

_C.eval = CN()
_C.eval.det_thresh = 0.01
_C.eval.gallery_size = 100
_C.eval.graph_thred = 0.0
_C.eval.nms_thresh = 0.5
_C.eval.use_data = ""
_C.eval.use_gt = False
_C.eval.device = "cuda"
_C.eval.dataset_file = "cuhk-sysu"
_C.eval.dataset_path = "data/cuhk-sysu"
_C.eval.eval_context = False
_C.eval.eval_all_sim = False
_C.eval.eval_method = "sim"  # proposed for future application.
_C.eval.checkpoint = "checkpoint.pth"
_C.eval.use_fast_graph = False
_C.eval.fast_graph_topk = 50

# -------------------------------------------------------- #
#                           Miscs                          #
# -------------------------------------------------------- #
# Directory where output files are written
_C.output_dir = "exps"
_C.seed = 42
_C.device = "cuda"


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
