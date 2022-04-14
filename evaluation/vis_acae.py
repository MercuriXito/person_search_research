""" modules providing visualization methods for ACAE model, including:
- attention scores in cross-attention module
- heatmap of features with Grad-CAM
"""
from genericpath import exists
import os
import cv2
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

import torchvision.ops.boxes as box_ops
import torchvision.transforms as T

from evaluation.eval_graph import build_and_load_from_dir
from datasets import load_eval_datasets
from models.graph_net import GraphNet
from utils.misc import ship_to_cuda
from utils.vis import draw_boxes_text, get_random_colors


NORMAL_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)


def min_max_rescale(x):
    return (x - x.min()) / (x.max() - x.min())


class GraphNetForward(nn.Module):
    """ Complete GraphNet forward, including extracting contextual features.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net
        assert isinstance(self.net, GraphNet)

    def forward(self, images: list):
        assert len(images) % 2 == 0, f"{len(images)} should be even number."
        assert isinstance(self.net, GraphNet)
        detections, _ = self.net(images)
        ind_features = [item["embeddings"] for item in detections]

        ctx_features = []
        for bidx in range(0, len(ind_features), 2):
            qfeats, gfeats = self.net.graph_head.graph_head.inference_features(
                ind_features[bidx], ind_features[bidx+1])
            ctx_features.append(qfeats)
            ctx_features.append(gfeats)
        return detections, ctx_features


# ---------------------------------------------------------------------
# Feature visualization with CAM methods.
# ---------------------------------------------------------------------
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers.
    adapted from https://github.com/jacobgil/pytorch-grad-cam/blob/3014beaf2877e621e686e5afe7f718c01f1a74d5/pytorch_grad_cam/activations_and_gradients.py#L1  # noqa
    extended to modules whose outputs are List[Tensor] or Tuple[Tensor].
    """
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)

        if isinstance(activation, torch.Tensor):
            saved = activation.cpu().detach()
        elif isinstance(activation, (list, tuple)):
            saved = [act.cpu().detach() for act in activation]
        self.activations.append(saved)

    def save_gradient(self, module, input, output):
        target_outputs = []
        if isinstance(output, torch.Tensor):
            if not hasattr(output, "requires_grad") or not output.requires_grad:
                # You can only register hooks on tensor requires grad.
                return
            target_outputs.append(output)
        elif isinstance(output, (list, tuple)):
            for out in output:
                if not hasattr(out, "requires_grad") or not out.requires_grad:
                    # You can only register hooks on tensor requires grad.
                    continue
                target_outputs.append(out)

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)

            if isinstance(grad, torch.Tensor):
                saved = [grad.cpu().detach()]
            elif isinstance(grad, (list, tuple)):
                saved = [act.cpu().detach() for act in grad]

            self.gradients = saved + self.gradients

        for out in target_outputs:
            out.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

    def release_activations(self):
        while len(self.activations) != 0:
            self.activations.pop()

    def release_gradients(self):
        while len(self.gradients) != 0:
            self.gradients.pop()


class FeatureMapVisualizer:
    """ Visualize the heatmap of extracted individual features
    """
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self._load_init()
        self._load_forward_args()
        self.mode = "eval"  # based on testset

        save_root = self.checkpoint_path + ".vis_featmap"
        self.save_root = os.path.join(save_root, "indv_feats")
        os.makedirs(self.save_root, exist_ok=True)

    def _load_init(self):
        """ load every necessary params from exp_dir
        """
        model, t_args = build_and_load_from_dir(self.exp_dir)
        # assert isinstance(model, GraphNet)

        eval_args = t_args.eval
        checkpoint_path = os.path.join(self.exp_dir, eval_args.checkpoint)
        device = torch.device(eval_args.device)
        dataset = load_eval_datasets(eval_args)
        model.to(device)
        model.eval()

        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = model
        self.args = t_args
        self.eval_args = eval_args

        # mapping
        self.gallery_name_to_idx_map = dict([(item["im_name"], idx) for idx, item in enumerate(self.dataset.roidb)])  # noqa
        self.query_name_to_idx_map = dict([(item["im_name"], idx) for idx, item in enumerate(self.dataset.probes)])  # noqa

    def _load_forward_args(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.min_size = 900
        self.max_size = 1500
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def _search_ind_in_boxes(self, boxes, target_box, debug=False):
        if isinstance(target_box, np.ndarray):
            target_box = torch.tensor(target_box).to(boxes)
        target_box = target_box.flatten()
        if len(target_box) == 5:
            target_box = target_box[:4]
        if boxes.shape[1] == 5:
            boxes = boxes[:, :4]
        target_box = target_box.view(1, 4)

        iou = box_ops.box_iou(boxes, target_box)
        ind = iou.flatten().argmax()

        if debug:
            # TODO: is this necessary?
            assert iou[ind] > 0.8, "Not Match!?"

        if iou[ind] <= 0.8:
            return ind.new_tensor(-1)  # -1
        return ind

    def vis_feature_map(
                    self,
                    query_inds: list,
                    gallery_inds: list,
                    query_boxes: list,
                    gallery_boxes: list,
                    ):

        self.model.to(self.device)
        self.model.eval()

        assert len(query_inds) == len(gallery_inds)
        assert len(query_boxes) == len(gallery_boxes)

        for qind, gind, qbox, gbox in \
                zip(query_inds, gallery_inds, query_boxes, gallery_boxes):

            # forward wrapper
            target_layers = [self.model.backbone.layer3]
            model = ActivationsAndGradients(self.model,
                                            target_layers=target_layers,
                                            reshape_transform=None)

            # load image and forward
            qitem = self.dataset.probes[qind]
            gitem = self.dataset.roidb[gind]
            target_boxes = [qbox, gbox]
            target_items = [qitem, gitem]

            images = [Image.open(item["path"]) for item in target_items]
            images = [self.transform(image) for image in images]
            images = ship_to_cuda(images, device=self.device)

            for image in images:
                image.requires_grad_(True)
            detections, _ = model(images)
            boxes, features, scores = [
                [out[key] for out in detections]
                for key in ["boxes", "embeddings", "scores"]
            ]

            for bidx, item in enumerate(target_items):
                # search target inds
                target_ind = self._search_ind_in_boxes(boxes[bidx], target_boxes[bidx])
                features[bidx][target_ind].sum().backward(retain_graph=True)  # BP

                # Grad-CAM Visualization
                feats, grads = model.activations, model.gradients

                # perform pixel-wise Grad-CAM for feature heatmap
                feats, grads = feats[0], grads[0]
                feats, grads = feats[bidx], grads[bidx]  # 3-d Tensor: [CxHxW]
                model.release_gradients()

                # channel_weights = torch.mean(grads, dim=[-1, -2], keepdim=True)
                # heatmap = torch.relu((feats * channel_weights).sum(dim=0))
                pixel_weights = torch.mean(torch.relu(grads), dim=0, keepdim=True)
                heatmap = torch.relu((pixel_weights * feats).sum(dim=0))

                heatmap = heatmap.detach().cpu().numpy()
                heatmap = min_max_rescale(heatmap)

                vised_heatmap = self.draw_heatmap(
                    item["path"],
                    heatmap,
                    boxes[bidx][target_ind].view(1, -1).detach().cpu().numpy()
                )
                saved_img_name = "pixel_heatmap_p{:02d}_{}".format(target_ind, item["im_name"])
                saved_img_path = os.path.join(self.save_root, saved_img_name)
                print(cv2.imwrite(saved_img_path, vised_heatmap))

    def draw_heatmap(self,
                     img_name: str,
                     heatmap: np.ndarray,
                     boxes: np.ndarray = None):
        """
        Args:
          - img_name: path of image
          - heatmap: mask
          - boxes: draw boxes
        """
        image = cv2.imread(img_name)
        height, width = image.shape[:2]

        if heatmap.dtype != np.uint8:
            heatmap = np.clip(heatmap, 0, 1) * 255.0
            heatmap = heatmap.astype(np.uint8)

        heatmap = cv2.resize(heatmap, dsize=(width, height))

        alpha = 0.6  # fade-in co-efficency.
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cimage = image * alpha + heatmap * (1-alpha)
        cimage = cimage.astype(np.uint8)

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(x) for x in box[:4]]
                cv2.rectangle(cimage, (x1, y1), (x2, y2), get_random_colors(i), thickness=2)

        return cimage


class CTXFeatureMapVisualizer(FeatureMapVisualizer):
    """ Visualize the heatmap of extracted ACAE features
    """
    def __init__(self, exp_dir):
        super().__init__(exp_dir)
        assert isinstance(self.model, GraphNet)
        self.model = GraphNetForward(self.model)

        save_root = self.checkpoint_path + ".vis_featmap"
        self.save_root = os.path.join(save_root, "ctx_feats")
        os.makedirs(self.save_root, exist_ok=True)

    def vis_feature_map(
                    self,
                    query_inds: list,
                    gallery_inds: list,
                    query_boxes: list,
                    gallery_boxes: list,
                    ):
        self.model.to(self.device)
        self.model.eval()

        assert len(query_inds) == len(gallery_inds)
        assert len(query_boxes) == len(gallery_boxes)

        for qind, gind, qbox, gbox in \
                zip(query_inds, gallery_inds, query_boxes, gallery_boxes):

            # forward wrapper
            target_layers = [
                self.model.net.backbone.layer3,
                # self.model.net.roi_heads.reid_embed_head,
                ]
            model = ActivationsAndGradients(self.model,
                                            target_layers=target_layers,
                                            reshape_transform=None)

            # load image and forward
            qitem = self.dataset.probes[qind]
            gitem = self.dataset.roidb[gind]
            target_boxes = [qbox, gbox]
            target_items = [qitem, gitem]

            images = [Image.open(item["path"]) for item in target_items]
            images = [self.transform(image) for image in images]
            images = ship_to_cuda(images, device=self.device)

            for image in images:
                image.requires_grad_(True)
            detections, ctx_features = model(images)
            boxes, features, scores = [
                [out[key] for out in detections]
                for key in ["boxes", "embeddings", "scores"]
            ]

            target_inds = []
            for bidx in range(len(target_items)):
                target_inds.append(
                    self._search_ind_in_boxes(boxes[bidx], target_boxes[bidx])
                )
            # check out if not matched
            if torch.any(torch.stack(target_inds) == -1):
                print(f"{qind}_{gind} matched: {torch.stack(target_inds).tolist()}")
                continue

            saved_img_root = os.path.join(self.save_root, f"{target_items[0]['im_name']}_{target_items[1]['im_name']}")
            os.makedirs(saved_img_root, exist_ok=True)
            ctx_notation = 0  # choose the final embeddings
            for bidx in range(len(target_items)):
                # vis on accord_bidx image, with bidx feature backward.

                accord_bidx = (bidx + 1) % 2  # vised on the other image
                item = target_items[accord_bidx]
                target_ind = target_inds[accord_bidx]

                # search target inds
                retain = False if bidx == len(target_items) - 1 else True
                ctx_features[bidx][ctx_notation][target_inds[bidx]].sum().backward(retain_graph=retain)  # BP

                # Grad-CAM Visualization
                feats, grads = model.activations, model.gradients

                # perform pixel-wise Grad-CAM for feature heatmap
                feats, grads = feats[0], grads[0]
                feats, grads = feats[accord_bidx], grads[accord_bidx]  # 3-d Tensor: [CxHxW]
                model.release_gradients()

                # channel_weights = torch.mean(grads, dim=[-1, -2], keepdim=True)
                # heatmap = torch.relu((feats * channel_weights).sum(dim=0))
                grads = torch.relu(grads)
                pixel_weights = torch.mean(grads, dim=0, keepdim=True)
                heatmap = torch.relu((pixel_weights * feats).sum(dim=0))

                heatmap = heatmap.detach().cpu().numpy()
                heatmap = min_max_rescale(heatmap)

                vised_heatmap = self.draw_heatmap(
                    item["path"], heatmap,
                    boxes[accord_bidx][target_ind].view(1, -1).detach().cpu().numpy()
                )

                # add boxes
                box = boxes[accord_bidx][target_ind]
                x1, y1, x2, y2 = [int(x) for x in box.flatten()[:4]]
                TARGET_COLOR = BLUE_COLOR if accord_bidx == 0 else GREEN_COLOR
                cv2.rectangle(vised_heatmap, (x1, y1), (x2, y2), TARGET_COLOR, thickness=2)

                saved_img_name = "pixel_acae_heatmap_p{:02d}_{}".format(target_ind, item["im_name"])
                saved_img_path = os.path.join(saved_img_root, saved_img_name)
                cv2.imwrite(saved_img_path, vised_heatmap)

                # save original image
                oimage = cv2.imread(target_items[bidx]["path"])
                x1, y1, x2, y2 = [int(x) for x in boxes[bidx][target_inds[bidx]].flatten()[:4]]
                cv2.rectangle(oimage, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
                saved_img_name = "pixel_acae_heatmap_p{:02d}_{}_from_{}".format(
                    target_ind, item["im_name"], target_items[bidx]["im_name"])
                saved_img_path = os.path.join(saved_img_root, saved_img_name)
                cv2.imwrite(saved_img_path, oimage)
            torch.cuda.empty_cache()
            model.release_activations()
            del model


# ---------------------------------------------------------------------
# Attention Weights Visualizatoin.
# ---------------------------------------------------------------------
class CTXAttnWeightsVisualizer(CTXFeatureMapVisualizer):
    """ Visualize the attn weights during features aggregation
    in ACAE module.
    """
    def __init__(self, exp_dir):
        super().__init__(exp_dir)
        self.weights = []
        self.register_attn_weights()

        self.save_root = self.checkpoint_path + ".vis_attn_weights"
        os.makedirs(self.save_root, exist_ok=True)

    def release_weights(self):
        while len(self.weights) != 0:
            self.weights.pop()

    def _save_attn_weights(self, module, input, output):
        """ design for nn.MultiheadAttention
        """
        _, weights = output
        self.weights.append(weights.detach().cpu())

    def register_attn_weights(self):
        for head in self.model.net.graph_head.graph_head.heads:
            head.self_attn.register_forward_hook(self._save_attn_weights)
            head.multihead_attn.register_forward_hook(self._save_attn_weights)

    def draw_attn_weights(
                    self,
                    query_inds: list,
                    gallery_inds: list,
                    query_boxes: list,
                    gallery_boxes: list,
                    ):

        self.model.to(self.device)
        self.model.eval()

        assert len(query_inds) == len(gallery_inds)
        assert len(query_boxes) == len(gallery_boxes)

        for qind, gind, qbox, gbox in \
                tqdm(zip(query_inds, gallery_inds, query_boxes, gallery_boxes)):

            # load image and forward
            qitem = self.dataset.probes[qind]
            gitem = self.dataset.roidb[gind]
            target_boxes = [qbox, gbox]
            target_items = [qitem, gitem]

            images = [Image.open(item["path"]) for item in target_items]
            images = [self.transform(image) for image in images]
            images = ship_to_cuda(images, device=self.device)

            with torch.no_grad():
                detections, _ = self.model(images)
            boxes = [out["boxes"] for out in detections]

            target_inds = []
            for bidx in range(len(target_items)):
                target_inds.append(
                    self._search_ind_in_boxes(boxes[bidx], target_boxes[bidx])
                )

            # check out if not matched
            if torch.any(torch.stack(target_inds) == -1):
                print(f"{qind}_{gind} matched: {torch.stack(target_inds).tolist()}")
                self.release_weights()
                continue

            saved_img_root = os.path.join(self.save_root, f"{qitem['im_name']}_{gitem['im_name']}")
            os.makedirs(saved_img_root, exist_ok=True)
            # HACK: due to the forward order, the order of obtained
            # attn_weights is different from the order of extracted
            # features, mind the order when refactor ctx modules.
            attn_inds = [[0, 1], [2, 3]]  # self and cross-attn weights for each samples.
            for bidx, (attn_ind, target_ind, item, boxes_img) in \
                        enumerate(zip(attn_inds, target_inds, target_items, boxes)):
                # self-attn: self-contained vis.
                self_weights = self.weights[attn_ind[0]][0][target_ind]
                cross_weights = self.weights[attn_ind[1]][0][target_ind]
                assert len(self_weights) == len(boxes_img)
                image = cv2.imread(item["path"])
                for pidx, box in enumerate(boxes_img):
                    x1, y1, x2, y2 = [int(x) for x in box.flatten()[:4]]
                    color = RED_COLOR if pidx == target_ind else NORMAL_COLOR
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
                    cv2.putText(image, f"{self_weights[pidx]:2.6f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                from_img_name = item["im_name"]
                saved_img_name = "self_attn_weights.{}".format(from_img_name)
                saved_img_path = os.path.join(saved_img_root, saved_img_name)
                cv2.imwrite(saved_img_path, image)

                # cross-attn: vis with two images
                accord_idx = (bidx + 1) % 2
                target_ind = target_inds[accord_idx]
                item = target_items[accord_idx]
                boxes_img = boxes[accord_idx]
                assert len(cross_weights) == len(boxes_img)
                image = cv2.imread(item["path"])
                for pidx, box in enumerate(boxes_img):
                    w = cross_weights[pidx]
                    if w < 0.05:
                        continue
                    x1, y1, x2, y2 = [int(x) for x in box.flatten()[:4]]
                    TARGET_COLOR = BLUE_COLOR if accord_idx == 0 else GREEN_COLOR
                    color = TARGET_COLOR if pidx == target_ind else NORMAL_COLOR
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
                    cv2.putText(image, f"{cross_weights[pidx]:2.6f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                to_img_name = item["im_name"]
                saved_img_name = "cross_attn_weights.{}_{}".format(from_img_name, to_img_name)
                saved_img_path = os.path.join(saved_img_root, saved_img_name)
                cv2.imwrite(saved_img_path, image)
            self.release_weights()  # release weights after processing each forward.


# ---------------------------------------------------------------------
# Test Functions.
# ---------------------------------------------------------------------
def random_samples(args, visualizer):
    from utils.misc import unpickle
    data = unpickle(args.pickle)
    eval_res = data["eval_rank"]["results"]
    print("Load pickle ok.")

    # random sample
    seed = 100
    random.seed(seed)

    qind = random.choice(list(range(len(eval_res))))
    entry = eval_res[qind]
    qimage, qbox = entry["probe_img"], entry["probe_roi"]

    correct_res = [item['correct'] for item in entry["gallery"]]
    if sum(correct_res) == 0:
        print("Suggest change a seed.")
        return

    idx = correct_res.index(1)
    gitem = entry["gallery"][idx]
    gimage, gbox = gitem['img'], gitem['roi'][:4]

    f_qind = visualizer.query_name_to_idx_map[qimage]
    f_gind = visualizer.gallery_name_to_idx_map[gimage]
    qbox = np.asarray(qbox)
    gbox = np.asarray(gbox)
    return f_qind, f_gind, qbox, gbox


def test():
    from easydict import EasyDict
    args = EasyDict(dict(
        exp_dir="exps/exps_acae/exps_cuhk.closs_35",
        # exp_dir="exps/exps_cuhk",
        pickle="exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl",
        seed=42,
    ))

    # visualizer = FeatureMapVisualizer(args.exp_dir)
    # visualizer = CTXFeatureMapVisualizer(args.exp_dir)
    visualizer = CTXAttnWeightsVisualizer(args.exp_dir)

    f_qind, f_gind, qbox, gbox = random_samples(args, visualizer)

    # fixed test samples.
    # f_qind, f_gind = 2619, 6382
    # qbox = np.asarray([29.0, 281.0, 93.0, 451.0])
    # gbox = np.asarray([134.69131469726562, 302.53204345703125, 188.76229858398438, 445.0682678222656])

    # visualizer.vis_feature_map(
    visualizer.draw_attn_weights(
        [f_qind],
        [f_gind],
        [qbox],
        [gbox]
    )


if __name__ == '__main__':
    test()
