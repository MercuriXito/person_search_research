import re
import os.path as osp
import random
from tqdm import tqdm

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
from numba import jit

from .ps_dataset import PersonSearchDataset
from utils.misc import pickle, unpickle, _compute_iou


class PRW(PersonSearchDataset):
    def __init__(self, root, transforms, mode):
        super().__init__(root, transforms, mode=mode)
        self.trainset_num_pids = 482

    def get_data_path(self):
        return osp.join(self.root, 'frames')

    def _load_image_set_index(self):
        if self.mode == 'train':
            imgs = loadmat(
                osp.join(self.root, 'frame_train.mat'))['img_index_train']
        elif self.mode in ('test', 'probe'):
            imgs = loadmat(
                osp.join(self.root, 'frame_test.mat'))['img_index_test']

        return [img[0][0] + '.jpg' for img in imgs]

    def gt_roidb(self):
        cache_file = osp.join(self.root, 'cache',
                              'PRW_{}_gt_roidb.pkl'.format(self.mode))

        if osp.isfile(cache_file):
            roidb = unpickle(cache_file)
            return roidb

        gt_roidb = []
        for im_name in self.imgs:
            anno_path = osp.join(self.root, 'annotations', im_name)
            anno = loadmat(anno_path)
            box_key = 'box_new'
            if box_key not in anno.keys():
                box_key = 'anno_file'
            if box_key not in anno.keys():
                box_key = 'anno_previous'

            rois = anno[box_key][:, 1:]
            ids = anno[box_key][:, 0]
            rois = np.clip(rois, 0, None)  # several coordinates are negative

            assert len(rois) == len(ids)

            rois[:, 2:] += rois[:, :2]
            gt_roidb.append({
                'im_name': im_name,
                'path': osp.join(self.get_data_path(), im_name),
                'boxes': rois.astype(np.int32),
                'gt_pids': ids.astype(np.int32),
                'flipped': False,
                'cam_id': self._get_cam_id(im_name)
            })

        # Search pairs
        label_filter = lambda x: np.where(x > 0)
        print("Search Pairs:")
        for i, item in enumerate(tqdm(gt_roidb)):
            pids = item["gt_pids"]
            keep_pid = label_filter(pids)
            pids = pids[keep_pid].reshape(-1, 1)

            # no labeled persons
            if pids.size == 0:
                pair_idx = random.randint(0, len(gt_roidb)-1)
                gt_roidb[i].update(pair_im_name=gt_roidb[pair_idx]["im_name"])
                continue

            # search overlap
            matches = []
            for j, pitem in enumerate(gt_roidb):
                if i == j:
                    matches.append(0)
                else:
                    ppids = pitem["gt_pids"]
                    keep_ppids = label_filter(ppids)
                    ppids = ppids[keep_ppids].reshape(1, -1)
                    num_equals = np.sum(pids == ppids).item()
                    matches.append(num_equals)
            matches = np.asarray(matches)
            max_num_overlap = np.max(matches)
            # randomly pick up all candidate
            candidate_indices = np.where(matches == max_num_overlap)[0]
            pair_idx = random.choice(candidate_indices)
            gt_roidb[i].update(pair_im_name=gt_roidb[pair_idx]["im_name"])

        pickle(gt_roidb, cache_file)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _adapt_pid_to_cls(self, label_pids, upid=5555):
        """
        convert pid range from (0, N-1) to (1, N), and replace -2 with unlabeled_person_identifier 5555
        """
        label_pids += 1
        label_pids += (label_pids == -1).type(torch.int64) * (upid + 1)
        return label_pids

    def load_probes(self):
        query_info = osp.join(self.root, 'query_info.txt')
        with open(query_info, 'r') as f:
            raw = f.readlines()

        probes = []
        for line in raw:
            linelist = line.split(' ')
            pid = int(linelist[0])
            x, y, w, h = float(linelist[1]), float(
                linelist[2]), float(linelist[3]), float(linelist[4])
            roi = np.array([x, y, x + w, y + h]).astype(np.int32)
            roi = np.clip(roi, 0, None)  # several coordinates are negative
            im_name = linelist[5][:-1] + '.jpg'
            probes.append({'im_name': im_name,
                           'path': osp.join(self.get_data_path(), im_name),
                           'boxes': roi[np.newaxis, :],
                           # Useless. Can be set to any value.
                           'gt_pids': np.array([pid]),
                           'flipped': False,
                           'cam_id': self._get_cam_id(im_name)})

        return probes

    def load_probes_with_ctx(self):
        query_info = osp.join(self.root, 'query_info.txt')
        with open(query_info, 'r') as f:
            raw = f.readlines()

        probes = []
        for line in raw:
            linelist = line.split()
            pid = int(linelist[0])
            # x, y, w, h = [float(x) for x in linelist[1:-1]]
            im_name = linelist[-1] + ".jpg"

            # annotation files including surrounding persons
            anno_path = osp.join(self.root, 'annotations', im_name)
            anno = loadmat(anno_path)
            box_key = 'box_new'
            if box_key not in anno.keys():
                box_key = 'anno_file'
            if box_key not in anno.keys():
                box_key = 'anno_previous'

            all_rois = anno[box_key][:, 1:]
            all_pids = anno[box_key][:, 0]
            all_rois = np.clip(all_rois, 0, None)  # several coordinates are negative
            all_rois[:, 2:] += all_rois[:, :2]

            # move the query target to the last one
            idx = np.where(all_pids == pid)[0][0]
            indices = np.arange(0, len(all_rois)).astype(np.int)
            indices[-1], indices[idx] = indices[idx], indices[-1]
            all_rois = all_rois[indices]
            all_pids = all_pids[indices]

            probes.append({
                'im_name': im_name,
                'path': osp.join(self.get_data_path(), im_name),
                'boxes': all_rois.astype(np.int),
                # Useless. Can be set to any value.
                'gt_pids': all_pids.astype(np.int),
                'flipped': False,
                'cam_id': self._get_cam_id(im_name)
            })

        return probes


    def _get_cam_id(self, im_name):
        match = re.search('c\d', im_name).group().replace('c', '')
        return int(match)

    @staticmethod
    @jit(forceobj=True)
    def search_performance_calc(gallery_set, probe_set,
                                gallery_det, gallery_feat, probe_feat,
                                det_thresh=0.5, gallery_size=-1, ignore_cam_id=True):
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
        for i in range(len(probe_set)):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0

            feat_p = probe_feat[i].ravel()

            probe_imname = probe_set.record[i]['im_name']
            probe_roi = probe_set.record[i]['boxes']
            probe_pid = probe_set.record[i]['gt_pids']
            probe_cam = probe_set.record[i]['cam_id']

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
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
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

        return ret

    def detection_performance_calc(
            self, gallery_det, det_thresh=0.5,
            iou_thresh=0.5, labeled_only=False):
        """
        gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image

        det_thresh (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """
        assert len(self.imgs) == len(gallery_det)
        gt_roidb = self.roidb

        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        for gt, det in zip(gt_roidb, gallery_det):
            gt_boxes = gt['boxes']
            if labeled_only:
                inds = np.where(gt['gt_pids'].ravel() > 0)[0]
                if len(inds) == 0:
                    continue
                gt_boxes = gt_boxes[inds]
            if det != []:
                det = np.asarray(det)
                inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
                det = det[inds]
                num_gt = gt_boxes.shape[0]
                num_det = det.shape[0]
            else:
                num_det = 0
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= iou_thresh)
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
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        # precision, recall, __ = precision_recall_curve(y_true, y_score)
        # recall *= det_rate

        print('{} detection:'.format(
            'labeled only' if labeled_only else 'all'))
        print('  recall = {:.2%}'.format(det_rate))
        if not labeled_only:
            print('  ap = {:.2%}'.format(ap))
        return ap, det_rate


if __name__ == '__main__':

    from datasets.transforms import get_transform

    root = "data/prw"
    transform = get_transform(True)
    data = PRW(root, transform, "train")

    from IPython import embed
    embed()
