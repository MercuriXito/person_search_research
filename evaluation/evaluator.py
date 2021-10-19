import numpy as np
import torch
import os.path as osp

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

from utils.misc import _compute_iou
from models.graph_net import GraphNet
from models.ctx_attn_head import ContextGraphHead


def sigmoid(arr):
    return 1 / (np.exp(-arr) + 1)


def graph_dist(image_match_scores, reid_scores):
    """ adaptive graph distance, excluding the context from gallery target.
    """
    return_scores = []
    for idx in range(len(reid_scores)):
        reid_score = reid_scores[idx, 0]
        valid_pos_mask = np.ones_like(image_match_scores)
        valid_pos_mask[idx, :] = 0
        valid_pos_mask = valid_pos_mask.astype(np.bool)
        valid_context = image_match_scores > reid_score
        valid_context = np.logical_and(valid_context, valid_pos_mask)
        if valid_context.sum() == 0:
            return_scores.append(float(reid_score))
        else:
            return_scores.append(float(image_match_scores[valid_context].mean()))
    return np.array(return_scores)[:, None]


def get_similarity_matrix_one_step_gpu(
        gallery_features, query_features, graph_thred,
        attn_func, device):
    """ all features and boxes are np.ndarray input.
    """
    if attn_func is not None:
        # init
        torch.cuda.set_device(device)
        attn_func.to(device)
        # tensor as input.
        gallery_features = [torch.as_tensor(feat) for feat in gallery_features]
        query_features = [torch.as_tensor(feat) for feat in query_features]

    gallery_similarity = []
    gallery_matches = []
    pbar = tqdm(total=len(gallery_features))
    for gallery_feat in gallery_features:
        pbar.update(1)
        query_sim, matches = [], []
        for query_feat in query_features:
            if gallery_feat.shape[0] == 0:
                query_sim.append(np.empty((0, 1)))
                break
            if attn_func is None:
                attn_func = get_context_sim
            sim = attn_func(gallery_feat, query_feat, graph_thred)
            query_sim.append(sim)
        sim = np.concatenate(query_sim, axis=1)
        assert sim.shape[0] == gallery_feat.shape[0]
        gallery_similarity.append(sim)
        gallery_matches.append(matches)

    return gallery_similarity, gallery_matches


def get_context_sim(
        gallery_feat, query_feat, graph_thred,
        *args, **kwargs):
    """ Context Similarity between features in query images and gallery images.
    """
    # HACK: the last ones in query features must be the target features.
    idx = -1

    query_context_feat = query_feat[:idx, :]
    query_target_feat = query_feat[idx, :][None]

    indv_scores = np.matmul(gallery_feat, query_target_feat.T)
    if len(query_target_feat) == 0:
        return indv_scores
    sim_matrix = np.matmul(gallery_feat, query_context_feat.T)

    # contextual scores
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    qg_mask = np.zeros_like(sim_matrix)
    qg_mask[row_ind, col_ind] = 1
    qg_sim_matrix = sim_matrix * qg_mask
    graph_scores = graph_dist(qg_sim_matrix, indv_scores)
    final_scores = indv_scores * (1 - graph_thred) + graph_scores * graph_thred

    # split scores
    final_scores = torch.as_tensor(final_scores)
    final_scores_softmax = torch.softmax(final_scores, 0)
    final_scores = final_scores_softmax * final_scores / final_scores_softmax.max()
    final_scores = np.array(final_scores.cpu())
    return final_scores


def get_cosine_sim(gallery_feat, query_feat):
    return np.matmul(gallery_feat, query_feat.T)


class PersonSearchEvaluator:
    def __init__(self, dataset_file="cuhk-sysu") -> None:
        self.dataset_file = dataset_file
        if self.dataset_file == "cuhk-sysu":
            self.eval_search = self.search_performance_by_sim
        elif self.dataset_file == "prw":
            self.eval_search = self.search_performance_by_sim_prw
        else:
            raise NotImplementedError(f"{dataset_file}")

    def get_similarity(
            self, gallery_feat, query_feat, use_context=True, graph_thred=0.0):
        if len(query_feat.shape) == 1:
            query_feat = query_feat.reshape(1, -1)
        if len(gallery_feat.shape) == 1:
            gallery_feat = gallery_feat.reshape(1, -1)

        if not use_context:
            query_target_feat = query_feat[-1].reshape(1, -1)
            return get_cosine_sim(gallery_feat, query_target_feat)
        return get_context_sim(gallery_feat, query_feat, graph_thred=graph_thred)

    def search_performance_by_sim(
            self, gallery_set, probe_set,
            gallery_det, gallery_feat, probe_feat, *,
            det_thresh=0.5, gallery_size=100,
            use_context=False, graph_thred=0.0, **kwargs):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image

        det_thresh (float): filter out gallery detections whose scores below this
        gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                            -1 for using full set
        """
        assert len(gallery_set) == len(gallery_det)
        assert len(gallery_set) == len(gallery_feat)
        assert len(probe_set) == len(probe_feat)

        use_context = use_context
        graph_thred = graph_thred

        use_full_set = gallery_size == -1
        fname = 'TestG{}'.format(gallery_size if not use_full_set else 50)
        protoc = loadmat(osp.join(gallery_set.root, 'annotation/test/train_test',
                                    fname + '.mat'))[fname].squeeze()

        # mapping from gallery image to (det, feat)
        gt_roidb = gallery_set.record
        name_to_det_feat = {}
        for gt, det, feat in zip(gt_roidb, gallery_det, gallery_feat):
            name = gt['im_name']
            if det != []:
                scores = det[:, 4].ravel()
                inds = np.where(scores >= det_thresh)[0]
                if len(inds) > 0:
                    # name_to_det_feat[name] = (det[inds], feat[inds])
                    gt_boxes = gt['boxes']
                    det_boxes, reID_feat_det = det[inds], feat[inds],
                    box_true = []
                    num_gt, num_det = gt_boxes.shape[0], det_boxes.shape[0]

                    # tag if detection is correct; could be skipped.
                    ious = np.zeros((num_gt, num_det), dtype=np.float32)
                    for i in range(num_gt):
                        for j in range(num_det):
                            ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
                    tfmat = (ious >= 0.5)
                    # for each det, keep only the largest iou of all the gt
                    for j in range(num_det):
                        largest_ind = np.argmax(ious[:, j])
                        for i in range(num_gt):
                            if i != largest_ind:
                                tfmat[i, j] = False
                    # for each gt, keep only the largest iou of all the det
                    for i in range(num_gt):
                        largest_ind = np.argmax(ious[i, :])
                        for j in range(num_det):
                            if j != largest_ind:
                                tfmat[i, j] = False
                    for j in range(num_det):
                        if tfmat[:, j].any():
                            box_true.append(True)
                        else:
                            box_true.append(False)

                    assert len(box_true) == len(det_boxes)
                    name_to_det_feat[name] = (
                        det_boxes, reID_feat_det, np.array(box_true))

        assert len(probe_set) > 0
        if "search_idx" in probe_set[0]:
            # if designate indices for searching.
            indices = [item["search_idx"] for item in probe_set]
        else:
            indices = range(len(probe_set))

        aps = []
        accs = []
        topk = [1, 5, 10]
        ret = {'image_root': gallery_set.data_path, 'results': []}
        for i, si in tqdm(enumerate(indices)):
            y_true, y_score, y_true_box = [], [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i]
            # Ignore the probe image
            probe_imname = str(protoc['Query'][si]['imname'][0, 0][0])
            probe_roi = protoc['Query'][si][
                'idlocate'][0, 0][0].astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probe_gt = []
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol
            for item in protoc['Gallery'][si].squeeze():
                gallery_imname = str(item[0][0])
                # some contain the probe (gt not empty), some not
                gt = item[1][0].astype(np.int32)
                count_gt += (gt.size > 0)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g, box_true = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # get similarity
                sim = self.get_similarity(
                    feat_g, feat_p, use_context=use_context,
                    graph_thred=graph_thred)
                sim = sim.flatten()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    w, h = gt[2], gt[3]
                    gt[2:] += gt[:2]
                    probe_gt.append({'img': str(gallery_imname),
                                        'roi': map(float, list(gt))})
                    iou_thresh = min(0.5, (w * h * 1.0) /
                                        ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    box_true = box_true[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                y_true_box.extend(list(box_true))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)
            # 2. Go through the remaining gallery images if using full set
            if use_full_set:
                for gallery_imname in gallery_set.imgs:
                    if gallery_imname in tested:
                        continue
                    if gallery_imname not in name_to_det_feat:
                        continue
                    det, feat_g, box_true = name_to_det_feat[gallery_imname]
                    # get L2-normalized feature matrix NxD
                    assert feat_g.size == np.prod(feat_g.shape[:2])
                    feat_g = feat_g.reshape(feat_g.shape[:2])
                    # get similarity
                    sim = self.get_similarity(
                        feat_g, feat_p, use_context=use_context,
                        graph_thred=graph_thred)
                    sim = sim.flatten()
                    # guaranteed no target probe in these gallery images
                    label = np.zeros(len(sim), dtype=np.int32)
                    y_true.extend(list(label))
                    y_score.extend(list(sim))
                    y_true_box.extend(list(box_true))
                    imgs.extend([gallery_imname] * len(sim))
                    rois.extend(list(det))
            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            y_true_box = np.asarray(y_true_box)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            y_true_box = y_true_box[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])
            # 4. Save result for JSON dump
            new_entry = {'probe_img': str(probe_imname),
                         'probe_roi': map(float, list(probe_roi)),
                         'probe_gt': probe_gt,
                         'probe_ap': ap,
                         'gallery': []}
            # only save top-10 predictions
            for k in range(10):
                new_entry['gallery'].append({
                    'img': str(imgs[inds[k]]),
                    'roi': map(float, list(rois[inds[k]])),
                    'score': float(y_score[k]),
                    'correct': int(y_true[k]),
                    'det_correct': int(y_true_box[k]),
                })
            ret['results'].append(new_entry)

        print('search ranking:')
        print('  mAP = {:.2%}'.format(np.mean(aps)))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print('  top-{:2d} = {:.2%}'.format(k, accs[i]))

        ret['mAP'] = np.mean(aps)
        ret['accs'] = accs

        mAP = np.mean(aps)
        top1 = accs[0]
        top5 = accs[1]
        top10 = accs[2]

        return mAP, top1, top5, top10, ret

    def search_performance_by_sim_prw(
                    self, gallery_set, probe_set,
                    gallery_det, gallery_feat, probe_feat, *,
                    det_thresh=0.5, gallery_size=-1, ignore_cam_id=True,
                    use_context=False, graph_thred=0.0, **kwargs):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image

        det_thresh (float): filter out gallery detections whose scores below this
        gallery_size (int): -1 for using full set
        ignore_cam_id (bool): Set to True acoording to CUHK-SYSU, 
                                alyhough it's a common practice to focus on cross-cam match only. 
        """
        assert len(gallery_set) == len(gallery_det)
        assert len(gallery_set) == len(gallery_feat)
        assert len(probe_set) == len(probe_feat)

        gt_roidb = gallery_set.record
        name_to_det_feat = {}
        for gt, det, feat in zip(gt_roidb, gallery_det, gallery_feat):
            name = gt['im_name']
            pids = gt['gt_pids']
            cam_id = gt['cam_id']
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds], pids, cam_id)

        aps = []
        accs = []
        topk = [1, 5, 10]
        ret = {'image_root': gallery_set.data_path, 'results': []}
        for i in tqdm(range(len(probe_set))):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0

            feat_p = probe_feat[i]
            probe_imname = probe_set[i]['im_name']
            probe_roi = probe_set[i]['boxes']
            probe_pid = probe_set[i]['gt_pids']
            probe_cam = probe_set[i]['cam_id']

            # Find all occurence of this probe
            gallery_imgs = []
            for x in gt_roidb:
                if probe_pid in x['gt_pids'] and x['im_name'] != probe_imname:
                    gallery_imgs.append(x)
            probe_gts = {}
            for item in gallery_imgs:
                probe_gts[item['im_name']] = \
                    item['boxes'][item['gt_pids'] == probe_pid]

            # Construct gallery set for this probe
            if ignore_cam_id:
                gallery_imgs = []
                for x in gt_roidb:
                    if x['im_name'] != probe_imname:
                        gallery_imgs.append(x)
            else:
                gallery_imgs = []
                for x in gt_roidb:
                    if x['im_name'] != probe_imname and x['cam_id'] != probe_cam:
                        gallery_imgs.append(x)

            # # 1. Go through all gallery samples
            # for item in testset.targets_db:
            # Gothrough the selected gallery
            for item in gallery_imgs:
                gallery_imname = item['im_name']
                # some contain the probe (gt not empty), some not
                count_gt += (gallery_imname in probe_gts)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g, _, _ = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # get similarity
                sim = self.get_similarity(
                    feat_g, feat_p, use_context=use_context,
                    graph_thred=graph_thred)
                sim = sim.flatten()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gallery_imname in probe_gts:
                    gt = probe_gts[gallery_imname].ravel()
                    w, h = gt[2] - gt[0], gt[3] - gt[1]
                    iou_thresh = min(0.5, (w * h * 1.0) /
                                        ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))

            # 2. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])
            # 4. Save result for JSON dump
            new_entry = {'probe_img': str(probe_imname),
                         'probe_roi': map(float, list(probe_roi.squeeze())),
                         'probe_gt': probe_gts,
                         'probe_ap': ap,
                         'gallery': []}
            # only save top-10 predictions
            for k in range(10):
                new_entry['gallery'].append({
                    'img': str(imgs[inds[k]]),
                    'roi': map(float, list(rois[inds[k]])),
                    'score': float(y_score[k]),
                    'correct': int(y_true[k]),
                })
            ret['results'].append(new_entry)

        print('search ranking:')
        mAP = np.mean(aps)
        print('  mAP = {:.2%}'.format(mAP))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print('  top-{:2d} = {:.2%}'.format(k, accs[i]))

        ret['mAP'] = np.mean(aps)
        ret['accs'] = accs

        top1 = accs[0]
        top5 = accs[1]
        top10 = accs[2]

        return mAP, top1, top5, top10, ret


class GraphPSEvaluator(PersonSearchEvaluator):
    """ Evaluator adapted for ACAE branch, which differs from original evaluator
    when getting the similarity.
    """
    def __init__(self, graph_head, device, dataset_file="cuhk-sysu", **eval_kwargs) -> None:
        super().__init__(dataset_file)
        self.graph_head = graph_head
        self.device = device
        assert isinstance(self.graph_head, ContextGraphHead)

        self.eval_all_sim = False
        if "eval_all_sim" in eval_kwargs and eval_kwargs["eval_all_sim"] == True:
            self.eval_all_sim = True
            print("Eval all sim.")

    def get_similarity(
            self, gallery_feat, query_feat, use_context, graph_thred,
            **eval_kwargs):
        if len(query_feat.shape) == 1:
            query_feat = query_feat.reshape(1, -1)
        if len(gallery_feat.shape) == 1:
            gallery_feat = gallery_feat.reshape(1, -1)

        if not use_context:
            query_target_feat = query_feat[-1].reshape(1, -1)
            return get_cosine_sim(gallery_feat, query_target_feat)

        idx = -1
        query_context_feat = query_feat[:idx, :]
        query_target_feat = query_feat[idx, :][None]

        indv_scores = np.matmul(gallery_feat, query_target_feat.T)
        if len(query_context_feat) == 0:
            return indv_scores

        with torch.no_grad():
            gallery_feat = torch.as_tensor(gallery_feat).to(self.device)
            query_context_feat = torch.as_tensor(query_context_feat).to(self.device)
            query_target_feat = torch.as_tensor(query_target_feat).to(self.device)
            scores = self.graph_head.inference(
                gallery_feat, query_context_feat, query_target_feat,
                graph_thred=graph_thred,
                eval_all_sim=self.eval_all_sim
            )
        scores = scores.cpu().numpy()
        return scores


class AggregatedPSEvaluator(PersonSearchEvaluator):
    """ Aggregated methods for computing similarity.
    """
    def __init__(
            self, graph_head, device,
            dataset_file="cuhk-sysu",
            dense_thresh=20) -> None:
        super().__init__(dataset_file)
        self.graph_head = graph_head
        self.device = device
        self.dense_thresh = dense_thresh
        assert isinstance(self.graph_head, ContextGraphHead)

    def get_similarity(self, gallery_feat, query_feat, use_context=True, graph_thred=0):
        num = len(query_feat)
        if num > self.dense_thresh:
            scores = self.get_graph_based_similarity(
                gallery_feat, query_feat, use_context, graph_thred)
        else:
            scores = super().get_similarity(
                gallery_feat, query_feat, use_context, graph_thred)
        return scores

    def get_graph_based_similarity(self, gallery_feat, query_feat, use_context=True, graph_thred=0):
        if len(query_feat.shape) == 1:
            query_feat = query_feat.reshape(1, -1)
        if len(gallery_feat.shape) == 1:
            gallery_feat = gallery_feat.reshape(1, -1)

        if not use_context:
            query_target_feat = query_feat[-1].reshape(1, -1)
            return get_cosine_sim(gallery_feat, query_target_feat)

        idx = -1
        query_context_feat = query_feat[:idx, :]
        query_target_feat = query_feat[idx, :][None]

        indv_scores = np.matmul(gallery_feat, query_target_feat.T)
        if len(query_context_feat) == 0:
            return indv_scores

        with torch.no_grad():
            gallery_feat = torch.as_tensor(gallery_feat).to(self.device)
            query_context_feat = torch.as_tensor(query_context_feat).to(self.device)
            query_target_feat = torch.as_tensor(query_target_feat).to(self.device)
            scores = self.graph_head.inference(
                gallery_feat, query_context_feat, query_target_feat,
                graph_thred=graph_thred
            )
        scores = scores.cpu().numpy()
        return scores


def build_evaluator(dataset, eval_method, model=None, device=None):
    if eval_method == "graph":
        assert model is not None
        assert device is not None
        assert isinstance(model, GraphNet)
        return GraphPSEvaluator(model.graph_head, device, dataset)
    elif eval_method == "aggregated":
        assert model is not None
        assert device is not None
        assert isinstance(model, GraphNet)
        return AggregatedPSEvaluator(model.graph_head, device, dataset)
    else:
        return PersonSearchEvaluator(dataset)
