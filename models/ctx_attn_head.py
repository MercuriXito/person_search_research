import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops

import random
from collections import defaultdict

from models.losses import CircleLoss, OIMLoss, ContrastiveLoss, TripletLoss


def reshape_back(x):
    return x.view(-1, x.size()[-1])


def normalize(x, dim=2):
    return F.normalize(x, p=2, dim=dim)


def apply_all(func, *args):
    res = []
    for arg in args:
        res.append(func(arg))
    return res


def mul(t1, t2):
    return torch.matmul(t1, t2.T)


def incremental_take(tensor, indices, dim):
    """ similar to take_along_dim
    """
    assert indices.ndim == 1
    assert tensor.ndim == 2, "only support 2-d tensor"

    increment_indices = torch.arange(len(indices)).to(indices)
    if dim == 0:
        return tensor[indices, increment_indices]
    else:
        return tensor[increment_indices, indices]


def similarity_to_distance(cos_sim):
    return (1 - cos_sim) / 2


def distance_to_similarity(dist):
    return 1 - 2 * dist


def build_criterion_for_graph_head(args):
    print(f"{args.name} for Graph Head.")
    if args.name == "contrastive":
        criterion = ContrastiveLoss(margin=args.margin)
    elif args.name == "triplet":
        criterion = TripletLoss(margin=args.margin)
    elif args.name == "circle":
        criterion = CircleLoss(m=args.margin, gamma=args.circle_gamma)
    else:
        criterion = OIMLoss(
            num_features=args.num_features,
            num_pids=args.num_pids,
            num_cq_size=args.num_cq_size,
            oim_momentum=args.oim_momentum,
            oim_scalar=args.oim_scalar,
        )
    return criterion


# ------------------------------ Graph Head ---------------------------------
def build_graph_head(*args, **kwargs):
    graph_module = DecoderGraph(*args, **kwargs)
    graph_head = ContextGraphHead(graph_module)
    return graph_head


class ContextGraphHead(nn.Module):
    """ GraphHead: for training and inference the graph structure.
    """
    def __init__(self, graph_head):
        super(ContextGraphHead, self).__init__()
        self.graph_head = graph_head

    def preprocess_det(self, detections, image_sizes=None,
                       clip_boxes=False, re_sample=False):
        """ preprocess the raw input of output proposals.
        All required input is in `List[Tensor]` format.
        args:
            - detections: List[Dict[str, Tensor]]: output of roi_heads
        """
        items = defaultdict(list)
        for det_item in detections:
            boxes = det_item["boxes"]  # N_anchors x num_class x 4
            scores = det_item["scores"]
            embeddings = det_item["embeddings"]
            pid_labels = det_item["pid_labels"]

            # select the person class.
            boxes = boxes[:, 1]
            scores = scores[:, 1]
            device = boxes.device

            # clip boxes
            if clip_boxes:
                assert image_sizes is not None
                image_height, image_width = image_sizes
                boxes = box_ops.clip_boxes_to_image(boxes, (image_height, image_width))

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores = boxes[keep], scores[keep]
            pid_labels, embeddings = pid_labels[keep], embeddings[keep]

            # valid det_thresh
            keep = torch.where(scores > 0.5)[0]
            boxes, scores = boxes[keep], scores[keep]
            pid_labels, embeddings = pid_labels[keep], embeddings[keep]

            # nms
            keep = box_ops.batched_nms(boxes, scores, pid_labels, iou_threshold=0.4)
            boxes, scores = boxes[keep], scores[keep]
            pid_labels, embeddings = pid_labels[keep], embeddings[keep]

            # valid labels: remove background and ignored samples
            labels_valid = (pid_labels > 0)
            boxes, scores = boxes[labels_valid], scores[labels_valid]
            pid_labels, embeddings = pid_labels[labels_valid], embeddings[labels_valid]

            if len(boxes) < 2 and re_sample:
                # 随机从 batch 中重采样
                num_person = len(det_item["boxes"])
                num_samples = 2 - len(boxes)
                add_inds = torch.LongTensor(
                    random.choices(list(range(num_person)), k=num_samples)
                ).to(device)

                boxes = torch.cat([boxes, det_item["boxes"][add_inds][1]])
                scores = torch.cat([scores, det_item["scores"][add_inds][:, 1:].flatten()])
                pid_labels = torch.cat([pid_labels, det_item["pid_labels"][add_inds]])
                embeddings = torch.cat([embeddings, det_item["embeddings"][add_inds]])

            items["boxes"].append(boxes)
            items["scores"].append(scores)
            items["pid_labels"].append(pid_labels)
            items["embeddings"].append(embeddings)

        return items["embeddings"], items["pid_labels"], \
            items["boxes"], items["scores"]

    def preprocess(
            self, embeddings, pid_labels, re_sample=False):
        """ preprocess for the output.
        All required input is in `List[Tensor]` format.
        args:
            - out_bbox: List[Tensor]. not in [0, 1] range.
        """
        items = defaultdict(list)
        for i, (pid_label, embedding) in \
                enumerate(zip(pid_labels, embeddings)):
            device = embedding.device

            # valid labels: remove background and ignored samples
            labels_valid = (pid_label > 0)
            pid_label, embedding = pid_label[labels_valid], embedding[labels_valid]

            if len(pid_label) < 2 and re_sample:
                # 随机从 batch 中重采样
                num_samples = 2 - len(pid_label)
                add_inds = torch.LongTensor(
                    random.choices(list(range(len(boxes[i]))), k=num_samples)
                ).to(device)
                pid_label = torch.cat([pid_label, pid_labels[i][add_inds]])
                embedding = torch.cat([embedding, embeddings[i][add_inds]])

            items["pid_labels"].append(pid_label)
            items["embeddings"].append(embedding)
        return items["embeddings"], items["pid_labels"]

    def forward(self, detections, targets, feats_lut, *args, **kwargs):
        """
        img_indices is required.
        """
        if self.training and feats_lut is None:
            return self.pair_forward(detections, targets)

        if self.training:
            assert feats_lut is not None

        # input from detections
        img_indices = [t["item_id"] for t in targets]
        embeddings, labels = [],  []
        for item in detections:
            embeddings.append(item["embeddings"])
            labels.append(item["pid_labels"])

        # test_embeddings, test_labels, _, _ = \
        #     self.preprocess_det(detections)

        # remove invalid persons
        embeddings, labels = self.preprocess(embeddings, labels)
        bs = len(embeddings)

        # find pairs
        pair_embeddings, pair_labels = \
            feats_lut.forward(img_indices, embeddings, labels)
        assert len(pair_embeddings) == bs

        # graph head forward
        outputs, losses = [], []
        for bidx in range(bs):
            output = self.graph_head(
                embeddings[bidx], pair_embeddings[bidx], labels[bidx], pair_labels[bidx])
            if self.training:
                scores, loss = output
                outputs.append(scores)
                losses.append(loss)
            else:
                outputs.append(output)
        loss_graph = torch.stack([l["graph_loss"] for l in losses])
        loss_graph = loss_graph.mean()
        losses = dict(loss_graph=loss_graph)
        return outputs, losses

    def pair_forward(self, detections, targets):
        embeddings, labels = [],  []
        for item in detections:
            embeddings.append(item["embeddings"])
            labels.append(item["pid_labels"])
        embeddings, labels = self.preprocess(embeddings, labels)
        bs = len(embeddings)
        outputs, losses = [], []
        for bidx in range(0, bs, 2):
            output = self.graph_head(
                embeddings[bidx], embeddings[bidx+1],
                labels[bidx], labels[bidx+1])
            if self.training:
                scores, loss = output
                outputs.append(scores)
                losses.append(loss)
            else:
                outputs.append(output)
        loss_graph = torch.stack([l["graph_loss"] for l in losses])
        loss_graph = loss_graph.mean()
        losses = dict(loss_graph=loss_graph)
        return outputs, losses

    def inference(
            self, gallery_embeddings,
            query_ctx_embedding,
            query_target_embedding,
            graph_thred,
            eval_all_sim=False):
        reid_scores = torch.matmul(gallery_embeddings, query_target_embedding.T)
        if len(query_ctx_embedding) == 0:
            return reid_scores

        gfeats = gallery_embeddings
        qfeats = torch.cat([query_ctx_embedding, query_target_embedding])
        sim_matrix = self.graph_head(
            gfeats, qfeats, eval_avg_sim=eval_all_sim)
        graph_scores = sim_matrix[:, -1][..., None]

        scores = graph_thred * graph_scores + (1 - graph_thred) * reid_scores
        sscores = torch.softmax(scores, dim=0)
        scores = sscores * scores / sscores.max()
        return scores


class DecoderGraph(nn.Module):
    def __init__(self, criterion, num_pids,
                 reid_feature_dim=256, num_stack=1,
                 nheads=4, dropout=0.0, *args, **kwargs):
        super().__init__()

        self.num_stack = num_stack
        self.feat_dim = reid_feature_dim
        # layer = nn.TransformerDecoderLayer
        layer = AttnGraphLayer
        self.heads = nn.ModuleList([
            layer(reid_feature_dim, nheads, reid_feature_dim, dropout=dropout)
            for i in range(self.num_stack)
        ])
        self.criterion = criterion
        # assert isinstance(self.criterion, OIMLoss)  # currently, only support OIMLoss
        self.num_pids = num_pids

    def construct_gt_mask(self, qlabels, glabels):
        """ GT matching result for training.
        as near-gt match.
        0: negative samples, 1: positive samples: other value: ignored samples.
        """

        # ground-truth match based on labels.
        m, n = len(qlabels), len(glabels)
        gt_mask = (qlabels.view(m, 1) == glabels.view(1, n)).type(torch.int)

        # for match between unlabeled persons, just ignored.
        q_ignored = (qlabels > self.num_pids)
        g_ignored = (glabels > self.num_pids)
        ignored_mask = torch.logical_and(
            q_ignored.view(m, 1), g_ignored.view(1, n)
        )
        gt_mask[ignored_mask] = -1  # ignored value
        return gt_mask

    def sample_pairs(self, sim_mat, qlabels, glabels):
        m, n = len(qlabels), len(glabels)
        match_mask = (qlabels.view(m, 1) == glabels.view(1, n))

        # for match between unlabeled persons, just ignored.
        q_ignored = (qlabels > self.num_pids)
        g_ignored = (glabels > self.num_pids)
        ignored_mask = torch.logical_and(
            q_ignored.view(m, 1), g_ignored.view(1, n)
        )
        pos_pair_mask = torch.logical_and(
            match_mask, torch.logical_not(ignored_mask)
        )
        neg_pair_mask = torch.logical_and(
            torch.logical_not(match_mask),
            torch.logical_not(ignored_mask)
        )
        pos_pair_mask = pos_pair_mask.type(torch.int)
        neg_pair_mask = neg_pair_mask.type(torch.int)

        # sample hard positive
        pos_values = (sim_mat * pos_pair_mask) + (1 - pos_pair_mask) * \
            (sim_mat.max(dim=1).values.unsqueeze(dim=1) + 1)
        pos_inds = torch.argmin(pos_values, dim=1)
        # sample hard negative
        neg_values = (sim_mat * neg_pair_mask) + (1 - neg_pair_mask) * \
            (-sim_mat.max(dim=1).values.unsqueeze(dim=1) - 1)
        neg_inds = torch.argmax(neg_values, dim=1)

        pos_score = incremental_take(sim_mat, pos_inds, 1)
        neg_score = incremental_take(sim_mat, neg_inds, 1)
        pos_valid = incremental_take(pos_pair_mask, pos_inds, 1)
        neg_valid = incremental_take(neg_pair_mask, neg_inds, 1)

        # mask invalid values
        pos_score = pos_score * pos_valid.to(pos_score)
        neg_score = neg_score * neg_valid.to(neg_score)
        return pos_score, neg_score

    def compute_loss_for_debug(
            self, sim_mat, qfeats, gfeats, qlabels, glabels):
        """ Only used for debug contrastive loss.
        """
        assert isinstance(self.criterion, ContrastiveLoss)
        # pair-wise losses which take pair samples as input.
        qpos_score, qneg_score = self.sample_pairs(
            sim_mat, qlabels, glabels)
        gpos_score, gneg_score = self.sample_pairs(
            sim_mat.T, glabels, qlabels)

        qloss = self.criterion(
            similarity_to_distance(qpos_score),
            similarity_to_distance(qneg_score))
        gloss = self.criterion(
            similarity_to_distance(gpos_score),
            similarity_to_distance(gneg_score))
        loss = (qloss + gloss) * 0.5
        return dict(
            qscores=(qpos_score, qneg_score),
            gscores=(gpos_score, gneg_score),
            qloss=qloss,
            gloss=gloss,
            graph_loss=loss,
        )

    def compute_loss(self, sim_mat, qfeats, gfeats, qlabels, glabels):
        if isinstance(self.criterion, (ContrastiveLoss, TripletLoss)):
            # pair-wise losses which take pair samples as input.
            qpos_score, qneg_score = self.sample_pairs(
                sim_mat, qlabels, glabels)
            gpos_score, gneg_score = self.sample_pairs(
                sim_mat.T, glabels, qlabels)

            qloss = self.criterion(
                similarity_to_distance(qpos_score),
                similarity_to_distance(qneg_score))
            gloss = self.criterion(
                similarity_to_distance(gpos_score),
                similarity_to_distance(gneg_score))
            loss = (qloss + gloss) * 0.5
        elif isinstance(self.criterion, (CircleLoss, )):
            # pair-wise losses which take pair samples as input.
            qpos_score, qneg_score = self.sample_pairs(
                sim_mat, qlabels, glabels)
            gpos_score, gneg_score = self.sample_pairs(
                sim_mat.T, glabels, qlabels)
            qloss = self.criterion(qpos_score / 2, qneg_score / 2)
            gloss = self.criterion(gpos_score / 2, gneg_score / 2)
            loss = (qloss + gloss) * 0.5
        elif isinstance(self.criterion, (OIMLoss, )):
            # feature-learning losses which take features and
            # labels as input.
            graph_feats = torch.cat([qfeats, gfeats])
            graph_labels = [qlabels, glabels]
            loss = self.criterion(graph_feats, graph_labels)
        else:
            raise Exception(f"Not Implemented Loss: \
                {str(self.criterion.__class__)}")
        return dict(graph_loss=loss)

    def forward(self, gfeats, qfeats,
                qlabels=None, glabels=None,
                eval_avg_sim=False, *args, **kwargs):
        """ [NxC] input for both features, return similarity matrix.
        """
        # reshape to sequence-like input
        qfeats = qfeats.view(-1, 1, self.feat_dim)
        gfeats = gfeats.view(-1, 1, self.feat_dim)

        # # HACK: detach configuration
        # qfeats = qfeats.detach()
        # gfeats = gfeats.detach()

        for layer in self.heads:
            qouts, gouts = layer(qfeats, gfeats), layer(gfeats, qfeats)
            qfeats, qself_feats, qcross_feats = qouts
            gfeats, gself_feats, gcross_feats = gouts

            # l2-normalize
            qfeats, qself_feats, qcross_feats = \
                apply_all(normalize, qfeats, qself_feats, qcross_feats)
            gfeats, gself_feats, gcross_feats = \
                apply_all(normalize, gfeats, gself_feats, gcross_feats)

        # reshape back
        qfeats, qself_feats, qcross_feats = \
            apply_all(reshape_back, qfeats, qself_feats, qcross_feats)
        gfeats, gself_feats, gcross_feats = \
            apply_all(reshape_back, gfeats, gself_feats, gcross_feats)

        if eval_avg_sim:
            sim_mat = torch.stack([
                torch.matmul(gfeats, qfeats.T),
                torch.matmul(gself_feats, qself_feats.T),
                torch.matmul(gcross_feats, qcross_feats.T)
            ], dim=0).mean(dim=0)
        else:
            sim_mat = torch.matmul(gfeats, qfeats.T)

        if self.training:
            assert qlabels is not None
            assert glabels is not None

            graph_loss = self.compute_loss(
                sim_mat, qfeats, gfeats, qlabels, glabels)
            return sim_mat, graph_loss
        return sim_mat

    def inference_features(self, qfeats, gfeats):

        # reshape to sequence-like input
        qfeats = qfeats.view(-1, 1, self.feat_dim)
        gfeats = gfeats.view(-1, 1, self.feat_dim)

        for layer in self.heads:
            qouts, gouts = layer(qfeats, gfeats), layer(gfeats, qfeats)
            qfeats, qself_feats, qcross_feats = qouts
            gfeats, gself_feats, gcross_feats = gouts

            # l2-normalize
            qfeats, qself_feats, qcross_feats = \
                apply_all(normalize, qfeats, qself_feats, qcross_feats)
            gfeats, gself_feats, gcross_feats = \
                apply_all(normalize, gfeats, gself_feats, gcross_feats)

        # reshape back
        qfeats, qself_feats, qcross_feats = \
            apply_all(reshape_back, qfeats, qself_feats, qcross_feats)
        gfeats, gself_feats, gcross_feats = \
            apply_all(reshape_back, gfeats, gself_feats, gcross_feats)

        return [qfeats, qself_feats, qcross_feats], \
            [gfeats, gself_feats, gcross_feats]


class AttnGraphLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0):
        super(AttnGraphLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt, memory):
        self_feat = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(self_feat)
        tgt = self.norm1(tgt)
        cross_feat = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(cross_feat)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_feat, cross_feat


# ------------------------------ Image LUT ---------------------------------
class ImageFeaturesLut:
    """ Lut for saving the temporal features
    """
    def __init__(self, dataset, feature_dim=256):
        self.dataset = dataset
        self.data = self.dataset.record
        self.imgs = dataset.imgs
        self.feature_dim = feature_dim
        self.epoch = 0

        self.cpu = torch.device("cpu")
        self.lut = dict()
        self._init_lut()

    def _init_lut(self):
        for i, item in enumerate(self.data):
            im_name = item["im_name"]
            gt_pids = torch.tensor(item["gt_pids"])
            gt_pids = self.dataset._adapt_pid_to_cls(gt_pids)
            pid_features = torch.zeros((len(gt_pids), self.feature_dim))
            self.lut.update({
                im_name: dict(
                    features=pid_features,
                    labels=gt_pids,
                )
            })

    def __len__(self):
        return len(self.lut)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, img_indices, features, pids):
        """ return features from paired image

        img_indices: List[Tensor], input image indices
        features: List[Tensor]
        pids: List[Tensor]
        rois: List[Tensor], box should be in size w.r.t original image,
                            in x1y1x2y2 format.
        """
        bs = len(img_indices)

        # foward: return the paried features
        pair_data = defaultdict(list)
        for bidx in range(bs):
            idx = img_indices[bidx].item()
            item = self.data[idx]
            pair_im_name = item["pair_im_name"]
            pair_info = self.lut[pair_im_name]

            pair_labels = pair_info["labels"].to(pids[bidx])
            pair_features = pair_info["features"].to(features[bidx])
            pair_data["labels"].append(pair_labels)
            pair_data["features"].append(pair_features)

        # backward: update the features in img_indices
        for bidx, (nlabels, new_features) in enumerate(zip(pids, features)):
            idx = img_indices[bidx].item()
            item = self.data[idx]
            im_name = item["im_name"]

            # update new features
            lut_info = self.lut[im_name]
            if len(nlabels) > 0:
                lut_info["features"] = new_features.detach().cpu()
                lut_info["labels"] = nlabels.detach().cpu()

        return pair_data["features"], pair_data["labels"]


class ImageEMAFeaturesLut:
    """ ImageLuT with EMA based feature update mechanisms.
    """
    def __init__(self, dataset, momentum=0.5, feature_dim=256, unlabel=5555):
        self.dataset = dataset
        self.data = self.dataset.record
        self.imgs = dataset.imgs
        self.feature_dim = feature_dim
        self.p_momentum = momentum
        self.momentum = self.p_momentum
        self.epoch = 0
        self.unlabel = unlabel

        self.cpu = torch.device("cpu")
        self.lut = dict()
        self._init_lut()

    def _init_lut(self):
        for i, item in enumerate(self.data):
            im_name = item["im_name"]
            gt_pids = torch.tensor(item["gt_pids"])
            gt_pids = self.dataset._adapt_pid_to_cls(gt_pids)
            pid_features = torch.zeros((len(gt_pids), self.feature_dim))
            # labels and rois would not be updated

            # 另一种更新方法：
            # labels
            # labeled features 使用 EMA 更新，在一个 epoch 中，由于 RandomCrop 增强，不一定所有的 pid 都会出现
            # unlabeled features 直接替换
            self.lut.update({
                im_name: dict(
                    labels=gt_pids.int(),
                    feats=pid_features,
                    unfeats=pid_features,
                )
            })

    def __len__(self):
        return len(self.lut)

    def set_epoch(self, epoch):
        if epoch == 0:
            self.epoch = 0
            self.momentum = 0.0
        else:
            self.epoch = epoch
            self.momentum = self.p_momentum

    def forward(self, img_indices, features, pids):
        """ return features from paired image

        img_indices: List[Tensor], input image indices
        features: List[Tensor]
        pids: List[Tensor]
        """
        bs = len(img_indices)

        # foward: return the paried features
        pair_data = defaultdict(list)
        for bidx in range(bs):
            device = features[bidx].device
            idx = img_indices[bidx].item()
            item = self.data[idx]
            pair_im_name = item["pair_im_name"]
            pair_info = self.lut[pair_im_name]

            pair_labels = pair_info["labels"]
            pair_feats = pair_info["feats"]
            pair_unfeats = pair_info["unfeats"].to(pair_feats)
            pair_unlabels = torch.full((len(pair_unfeats), ), self.unlabel).to(pair_labels)

            out_labels = torch.cat([pair_labels, pair_unlabels]).to(pids[bidx])
            out_feats = torch.cat([pair_feats, pair_unfeats]).to(features[bidx])

            pair_data["labels"] = out_labels
            pair_data["features"] = out_feats

        # backward: update the features in img_indices
        # label: EMA: 甚至可以用多个相同 pid 的 features 的平均更新
        # unlabel: cq
        eps = 1e-8
        for bidx, (nlabels, nfeats) in enumerate(zip(pids, features)):
            nlabels, nfeats = nlabels.to(self.cpu), nfeats.to(self.cpu)
            idx = img_indices[bidx].item()
            item = self.data[idx]
            im_name = item["im_name"]

            lut_info = self.lut[im_name]
            gt_pids = lut_info["labels"]
            t_feats = lut_info["feats"]

            # update labeled
            equals = gt_pids[..., None] == nlabels[None]
            valid = (torch.sum(equals, dim=1) > 0).type(torch.int)
            r_momentum = torch.as_tensor(1 - self.momentum).view(1).repeat(len(valid))
            r_momentum = r_momentum * valid.type(torch.float)
            r_momentum = r_momentum.view(-1, 1)

            matched_feats = \
                torch.matmul(equals.float(), nfeats) / (torch.sum(equals, dim=1, keepdim=True) + eps)
            feats = t_feats * (1 - r_momentum) + matched_feats * r_momentum

            lut_info["feats"] = feats

            # update unlabeled
            indices = torch.where(nlabels == self.unlabel)[0]
            if len(indices) == 0:
                continue
            unfeats = nfeats[indices]
            lut_info["unfeats"] = unfeats

        return pair_data["features"], pair_data["labels"]


if __name__ == '__main__':

    from datasets import build_trainset

    data_root = 'data/cuhk-sysu'
    image_set = 'train'
    data_name = "cuhk-sysu"
    dataset = build_trainset("cuhk-sysu", data_root)

    lut = ImageFeaturesLut(dataset)
    # lut = ImageEMAFeaturesLut(dataset)

    device = torch.device("cpu")

    # form input for testing
    n = 4
    img_indices = []
    features, plabels, rois = [], [], []
    for i in range(n):
        _, item = dataset[i]
        print(dataset.record[i]["im_name"])
        boxes = torch.as_tensor(dataset.record[i]["boxes"]).float().to(device)
        labels = torch.as_tensor(item["pid_labels"]).to(device)
        feats = torch.randn((len(labels), 256)).to(device)
        ind = torch.as_tensor(item["item_id"]).to(device)

        img_indices.append(ind)
        features.append(feats)
        plabels.append(labels)
        rois.append(boxes)

    res = lut.forward(
        img_indices, features, plabels
    )

    from IPython import embed
    embed()
