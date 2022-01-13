import numpy as np
import random
from numpy.core.numeric import cross
import torch
from datasets import load_eval_datasets
from datasets.cuhk_sysu import CUHK_SYSU
from evaluation.eval import FasterRCNNExtractor, evaluate
from evaluation.eval_defaults import build_and_load_from_dir
from evaluation.evaluator import GraphPSEvaluator, PseudoGraphEvaluator
import PIL.Image as Image
import cv2

import os
import os.path as osp
from utils.misc import unpickle
from easydict import EasyDict
from scipy.io import loadmat

from utils.vis import draw_boxes_text


class SPseudoGraphEvaluator(GraphPSEvaluator):
    def get_similarity(
            self, gallery_feat, query_feat, use_context, graph_thred, **eval_kwargs):

        if len(query_feat.shape) == 1:
            query_feat = query_feat.reshape(1, -1)
        if len(gallery_feat.shape) == 1:
            gallery_feat = gallery_feat.reshape(1, -1)

        idx = -1
        query_context_feat = query_feat[:idx, :]
        query_target_feat = query_feat[idx, :][None]
        indv_scores = np.matmul(gallery_feat, query_target_feat.T)

        qfeats, gfeats = query_feat, gallery_feat

        from IPython import embed
        embed()

        # example for producing the ground-truth
        cross_sim = np.matmul(qfeats, gfeats.T)
        M, N = cross_sim.shape

        ginds = cross_sim.argmax(axis=1)
        req_feats = gfeats[ginds]
        q_attn_weights = np.matmul(qfeats.reshape(M, 1, -1), req_feats.reshape(M, -1, 1))
        q_attn_weights = q_attn_weights.reshape(M, 1)
        q_agg_feats = np.mean(q_attn_weights * qfeats, axis=0, keepdims=True)

        qinds = cross_sim.argmax(axis=0)
        reg_feats = qfeats[qinds]
        g_attn_weights = np.matmul(gfeats.reshape(N, 1, -1), reg_feats.reshape(N, -1, 1))
        g_attn_weights = g_attn_weights.reshape(N, 1)
        g_agg_feats = np.mean(g_attn_weights * gfeats, axis=0, keepdims=True)

        cross_sim = np.matmul(q_agg_feats, g_agg_feats.T).flatten()

        final_scores = indv_scores * (1 - graph_thred) + cross_sim * graph_thred

        # split scores
        final_scores = torch.as_tensor(final_scores)
        final_scores_softmax = torch.softmax(final_scores, 0)
        final_scores = final_scores_softmax * final_scores / final_scores_softmax.max()
        final_scores = np.array(final_scores.cpu())
        return final_scores



def select_samples(dataset: CUHK_SYSU, seed=42):
    """ select random triple pair (query, pos_gallery, neg_gallery).
    """
    seed = 50
    random.seed(seed)

    gallery_size = 100
    use_full_set = gallery_size == -1
    fname = 'TestG{}'.format(gallery_size if not use_full_set else 50)
    protoc = loadmat(osp.join(dataset.root, 'annotation/test/train_test',
                                fname + '.mat'))[fname].squeeze()

    probes = dataset.probes
    num_query = len(probes)

    # load random query infos
    query_idx = random.randint(0, num_query-1)
    probe_roi = protoc['Query'][query_idx][
        'idlocate'][0, 0][0].astype(np.int32)
    probe_roi[2:] += probe_roi[:2]

    # choose samples from designated gallery
    gallerys = protoc["Gallery"][query_idx].squeeze()
    gt_indices = []
    other_indices = []

    for gidx, item in enumerate(gallerys):
        gt = item[1][0].astype(np.int32)
        if gt.size > 0:  # gt gallery
            gt_indices.append(gidx)
        else:
            other_indices.append(gidx)

    # pos idx and neg idx
    assert len(gt_indices) > 0
    assert len(other_indices) > 0

    pos_gidx = random.choice(gt_indices)
    neg_gidx = random.choice(other_indices)
    pos_name = str(gallerys[pos_gidx][0][0])
    neg_name = str(gallerys[neg_gidx][0][0])

    # search true indices in gallery
    pos_gidx, neg_gidx = -1, -1
    for idx, gitem in enumerate(dataset.roidb):
        if gitem["im_name"] == pos_name:
            pos_gidx = idx
        elif gitem["im_name"] == neg_name:
            neg_gidx = idx

    assert pos_gidx != -1
    assert neg_gidx != -1
    return query_idx, pos_gidx, neg_gidx


if __name__ == '__main__':

    exps_dir = os.path.join("exps/exps_cuhk.graph.lossw_101")
    pkl_path = os.path.join(exps_dir, "checkpoint.pth.acae.G0.4.eval.pkl")
    eval_path = os.path.join(exps_dir, "acae_eval.yml")

    # load trained pickle
    res_pkl = unpickle(pkl_path)
    query_features = res_pkl["query_features"]
    gallery_features = res_pkl["gallery_features"]
    gallery_boxes = res_pkl["gallery_boxes"]
    query_boxes = res_pkl["query_boxes"]

    # select samples
    dataset = load_eval_datasets(EasyDict(
        dataset_file="cuhk-sysu",
        root="dadta/cuhk-sysu/"
    ))
    query_idx, pos_idx, neg_idx = select_samples(dataset)

    # load models
    net, args = build_and_load_from_dir(exps_dir, eval_path)
    device = torch.device(args.device)
    # evaluator = GraphPSEvaluator(net.graph_head, device)
    net.eval()

    eval_args = args.eval
    extractor = FasterRCNNExtractor(net, device)
    # ps_evaluator = PseudoGraphEvaluator(
    #     net.graph_head, device, eval_args.dataset_file,
    #     eval_all_sim=eval_args.eval_all_sim)
    # ps_evaluator = GraphPSEvaluator(net.graph_head, device)
    ps_evaluator = SPseudoGraphEvaluator(
        net.graph_head, device, eval_args.dataset_file,
        eval_all_sim=eval_args.eval_all_sim)

    res_pkl, table_string = evaluate(extractor, eval_args, ps_evaluator=ps_evaluator, res_pkl=res_pkl)
    querys = dataset.probes
    gallerys = dataset.roidb

    exit(0)

    tbox = query_boxes[query_idx]
    pboxes = [gallery_boxes[pos_idx], gallery_boxes[neg_idx]]
    pprefix = ["pos", "neg"]

    # query image
    indices_text = [f"{i}" for i in range(len(tbox))]
    img_path = querys[query_idx]["path"]
    qimage = np.asarray(Image.open(img_path))
    vis_res = draw_boxes_text(qimage, tbox, str_texts=indices_text)
    vis_res = cv2.cvtColor(vis_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite("exps/vis/visualize/query.png", vis_res)

    # gallery image
    for i, idx in enumerate([pos_idx, neg_idx]):
        pbox = pboxes[i]
        prefix = pprefix[i]
        indices_text = [f"{i}" for i in range(len(pbox))]
        img_path = gallerys[idx]["path"]
        gimage = np.asarray(Image.open(img_path))
        vis_res = draw_boxes_text(gimage, pbox, str_texts=indices_text)
        vis_res = cv2.cvtColor(vis_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"exps/vis/visualize/gallery_{prefix}.png", vis_res)

    # visualize bounding boxes

    qfeats = query_features[query_idx]
    gfeats = gallery_features[pos_idx]
    neg_gfeats = gallery_features[neg_idx]

    new_sim = net.graph_head.graph_head.forward(
        torch.as_tensor(qfeats).to(device),
        torch.as_tensor(gfeats).to(device)
    )

    neg_sim = net.graph_head.graph_head.forward(
        torch.as_tensor(qfeats).to(device),
        torch.as_tensor(neg_gfeats).to(device)
    )
