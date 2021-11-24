import os
import cv2
import torch
import torch.nn as nn

from datasets import load_eval_datasets
from evaluation.badcase import get_args
from evaluation.eval_defaults import build_and_load_from_dir
from models.ctx_attn_head import AttnGraphLayer, DecoderGraph, \
    ContextGraphHead, apply_all, reshape_back, normalize
from models.graph_net import GraphNet
from utils.vis import draw_boxes_text
from utils.misc import unpickle


class VisAttnGraphLayer(AttnGraphLayer):
    """ AttnGraphLayer for visualization.
    """
    def forward(self, tgt, memory):
        self_feat, self_attn = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(self_feat)
        tgt = self.norm1(tgt)
        cross_feat, cross_attn = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(cross_feat)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_feat, cross_feat, self_attn, cross_attn


class VisDecoderGraph(DecoderGraph):
    """ DecoderGraph for visualization.
    """
    def __init__(self, criterion, num_pids,
                 reid_feature_dim=256, num_stack=1,
                 nheads=4, dropout=0.0, *args, **kwargs):
        super().__init__(
            criterion, num_pids, reid_feature_dim, num_stack,
            nheads, dropout, *args, **kwargs)
        layer = VisAttnGraphLayer
        self.heads = nn.ModuleList([
            layer(reid_feature_dim, nheads, reid_feature_dim, dropout=dropout)
            for i in range(self.num_stack)
        ])

    def forward(self, gfeats, qfeats,
                eval_avg_sim=False, *args, **kwargs):
        """ [NxC] input for both features, return similarity matrix.
        """
        assert not self.training
        # reshape to sequence-like input
        qfeats = qfeats.view(-1, 1, self.feat_dim)
        gfeats = gfeats.view(-1, 1, self.feat_dim)

        attns = []
        for layer in self.heads:
            qouts, gouts = layer(qfeats, gfeats), layer(gfeats, qfeats)
            qfeats, qself_feats, qcross_feats, qself_attn, qcross_attn \
                = qouts
            gfeats, gself_feats, gcross_feats, gself_attn, gcross_attn \
                = gouts
            attns.append(qself_attn)
            attns.append(qcross_attn)
            attns.append(gself_attn)
            attns.append(gcross_attn)

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
        return sim_mat, attns


class VisContextGraph(ContextGraphHead):
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
        sim_matrix, attns = self.graph_head(
            gfeats, qfeats, eval_avg_sim=eval_all_sim)
        graph_scores = sim_matrix[:, -1][..., None]

        scores = graph_thred * graph_scores + (1 - graph_thred) * reid_scores
        sscores = torch.softmax(scores, dim=0)
        scores = sscores * scores / sscores.max()
        return scores, attns


def sub_vis_layers(model):
    """ replace ACAE part for visualization.
    """
    assert isinstance(model, GraphNet)

    graph_head = model.graph_head

    vis_graph_module = VisDecoderGraph(
        graph_head.graph_head.criterion,
        graph_head.graph_head.num_pids,
        graph_head.graph_head.feat_dim,
        graph_head.graph_head.num_stack,
    )
    vis_graph_head = VisContextGraph(vis_graph_module)
    vis_graph_head.load_state_dict(graph_head.state_dict())

    model.graph_head = vis_graph_head
    return model


def select_samples(imdb):
    # query_name = "s483.jpg"
    # gallery_name = "s482.jpg"

    query_name = "s1259.jpg"
    gallery_name = "s1260.jpg"

    query_item = None
    qidx = None
    for idx, item in enumerate(imdb.probes):
        if item["im_name"] == query_name:
            query_item = item
            qidx = idx
            break

    gallery_item = None
    gidx = None
    for idx, item in enumerate(imdb.roidb):
        if item["im_name"] == gallery_name:
            gallery_item = item
            gidx = idx
            break

    assert query_item is not None
    assert gallery_item is not None
    return query_item, gallery_item, qidx, gidx


def main():
    args = get_args()
    pkl_path = args.pkl
    eval_data = unpickle(pkl_path)
    eval_args = eval_data["eval_args"]
    save_root = pkl_path + ".vis_attn"
    os.makedirs(save_root, exist_ok=True)

    device = torch.device(eval_args.device)
    imdb = load_eval_datasets(eval_args)
    img_root = imdb.get_data_path()

    # build model
    dirname = os.path.dirname(pkl_path)
    model, _ = build_and_load_from_dir(dirname)
    model = sub_vis_layers(model)
    model.to(device)
    model.eval()

    # select samples
    qitem, gitem, qidx, gidx = select_samples(imdb)

    # boxes and features
    query_boxes = eval_data["query_boxes"]
    gallery_boxes = eval_data["gallery_boxes"]
    query_feats = eval_data["query_features"]
    gallery_feats = eval_data["gallery_features"]

    qboxes = query_boxes[qidx]
    gboxes = gallery_boxes[gidx]
    qfeats = torch.tensor(query_feats[qidx]).to(device)
    gfeats = torch.tensor(gallery_feats[gidx]).to(device)
    qfeats = qfeats.view(-1, 256)
    gfeats = gfeats.view(-1, 256)

    query_context_feat = qfeats[:-1, :]
    query_target_feat = qfeats[-1, :][None]

    sim_mat, attns = model.graph_head.inference(
        gfeats, query_context_feat, query_target_feat,
        graph_thred=eval_args.graph_thred,
        eval_all_sim=True
    )
    sub_save_root = os.path.join(
        save_root, qitem["im_name"] + "_" + gitem["im_name"])
    os.makedirs(sub_save_root, exist_ok=True)

    # draw pictures
    gimg = cv2.imread(os.path.join(img_root, gitem["im_name"]))
    qimg = cv2.imread(os.path.join(img_root, qitem["im_name"]))

    gself_attn = attns[2][0][0].tolist()
    attn_str = ["{:2.4f}".format(x) for x in gself_attn]
    img = draw_boxes_text(gimg, gboxes, attn_str)
    cv2.imwrite(os.path.join(sub_save_root, "vis_gself_attn.png"), img)

    qcross_attn = attns[1][0][0].tolist()
    attn_str = ["{:2.4f}".format(x) for x in qcross_attn]
    img = draw_boxes_text(gimg, gboxes, attn_str)
    cv2.imwrite(os.path.join(sub_save_root, "vis_qcross_attn.png"), img)

    qself_attn = attns[0][0][0].tolist()
    attn_str = ["{:2.4f}".format(x) for x in qself_attn]
    img = draw_boxes_text(qimg, qboxes, attn_str)
    cv2.imwrite(os.path.join(sub_save_root, "vis_qself_attn.png"), img)

    gcross_attn = attns[3][0][0].tolist()
    attn_str = ["{:2.4f}".format(x) for x in gcross_attn]
    img = draw_boxes_text(qimg, qboxes, attn_str)
    cv2.imwrite(os.path.join(sub_save_root, "vis_gcross_attn.png"), img)


if __name__ == '__main__':
    main()
