import os.path as osp
import random
from tqdm import tqdm
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
from collections import defaultdict
from numba import jit

from .ps_dataset import PersonSearchDataset
from utils.misc import pickle, unpickle, _compute_iou


class CUHK_SYSU(PersonSearchDataset):
    def __init__(self, root, transforms, mode):
        super().__init__(root, transforms, mode=mode)
        self.trainset_num_pids = 5532

    def get_data_path(self):
        return osp.join(self.root, 'Image', 'SSM')

    def gt_roidb(self):
        cache_file = osp.join(self.root, 'cache',
                              'CUHK-SYSU_{}_gt_roidb.pkl'.format(self.mode))

        if osp.isfile(cache_file):
            roidb = unpickle(cache_file)
            return roidb

        # Load all images and build a dict from image to boxes
        all_imgs = loadmat(
            osp.join(self.root, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        for im_name, _, boxes in all_imgs:
            im_name = str(im_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, \
                'Warning: {} has no valid boxes.'.format(im_name)
            boxes = boxes[valid_index]
            name_to_boxes[im_name] = boxes.astype(np.int32)
            name_to_pids[im_name] = -1 * \
                np.ones(boxes.shape[0], dtype=np.int32)

        def _set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return
            print('Warning: person {} box {} cannot find in Images'.format(pid, box))

        # Load all the train/probe/test persons and number their pids from 0 to N-1
        # Background people have pid == -1
        if self.mode == 'train':
            train = loadmat(osp.join(self.root,
                                     'annotation/test/train_test/Train.mat'))
            train = train['Train'].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for im_name, box, __ in scenes:
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)
        else:
            test = loadmat(osp.join(self.root,
                                    'annotation/test/train_test/TestG50.mat'))
            test = test['TestG50'].squeeze()
            for index, item in enumerate(test):
                # query
                im_name = str(item['Query'][0, 0][0][0])
                box = item['Query'][0, 0][1].squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box,
                             name_to_pids[im_name], index)
                # gallery
                gallery = item['Gallery'].squeeze()
                for im_name, box, __ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)

        # Construct the gt_roidb
        gt_roidb = []
        for im_name in self.imgs:
            path = osp.join(self.get_data_path(), im_name)
            boxes = name_to_boxes[im_name]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]  # (x1, y1, x2, y2)
            pids = name_to_pids[im_name]
            gt_roidb.append({
                'im_name': im_name,
                'path': path,
                'boxes': boxes,
                'gt_pids': pids,
                'flipped': False})

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

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(osp.join(self.root, 'annotation', 'pool.mat'))
        test = test['pool'].squeeze()
        test = [str(a[0]) for a in test]
        if self.mode in ('test', 'probe'):
            return test
        # all images
        all_imgs = loadmat(
            osp.join(self.root, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training
        return list(set(all_imgs) - set(test))

    def _adapt_pid_to_cls(self, label_pids, upid=5555):
        """
        convert pid range from (0, N-1) to (1, N), and replace -1 with unlabeled_person_identifier 5555
        """
        label_pids += 1
        label_pids += (label_pids == 0).type(torch.int64) * upid
        return label_pids

    def load_probes(self):
        protoc = loadmat(osp.join(self.root,
                                  'annotation/test/train_test/TestG50.mat'))['TestG50'].squeeze()
        probes = []
        for item in protoc['Query']:
            im_name = str(item['imname'][0, 0][0])
            path = osp.join(self.get_data_path(), im_name)
            roi = item['idlocate'][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            probes.append({'im_name': im_name,
                           'path': path,
                           'boxes': roi[np.newaxis, :],
                           'gt_classes': np.array([1]),
                           # Useless. Can be set to any value.
                           'gt_pids': np.array([-100]),
                           'flipped': False})
        return probes

    def load_test_context_annotations(self):
        anno_path = osp.join(self.root, "annotation", "Person.mat")
        mat = loadmat(anno_path)
        mat = mat['Person'].squeeze()

        test_imgs = osp.join(self.root, "annotation", "pool.mat")
        test_imgs = loadmat(test_imgs)
        test_imgs = test_imgs['pool'].squeeze()
        test_imgs = [x[0] for x in test_imgs]
        test_imgs = set(test_imgs)

        targets = defaultdict(list)
        for item in mat:
            pid, n, locations = item
            locations = locations.squeeze()
            find_test_imgs = []
            for loc in locations:
                img_name, box_loc, ishard = loc
                img_name = img_name[0]
                real_box = box_loc.copy().astype(np.int)
                real_box[0][2] += real_box[0][0]
                real_box[0][3] += real_box[0][1]
                if img_name in test_imgs:
                    find_test_imgs.append([
                        img_name, pid[0], real_box
                    ])

            for fimg, pid, box_loc in find_test_imgs:
                targets[fimg].append([pid, box_loc])
        return targets

    @staticmethod
    @jit(forceobj=True)
    def search_performance_calc(gallery_set, probe_set,
                                gallery_det, gallery_feat, probe_feat,
                                det_thresh=0.5, gallery_size=100):
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

        aps = []
        accs = []
        topk = [1, 5, 10]
        ret = {'image_root': gallery_set.data_path, 'results': []}
        for i in range(len(probe_set)):
            y_true, y_score, y_true_box = [], [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            probe_imname = str(protoc['Query'][i]['imname'][0, 0][0])
            probe_roi = protoc['Query'][i][
                'idlocate'][0, 0][0].astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probe_gt = []
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol
            for item in protoc['Gallery'][i].squeeze():
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
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
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
                    # compute cosine similarities
                    sim = feat_g.dot(feat_p).ravel()
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

    root = "data/cuhk-sysu"
    transform = get_transform(True)
    data = CUHK_SYSU(root, transform, "train")

    from IPython import embed
    embed()
