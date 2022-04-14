import os
import cv2
import numpy as np
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from utils.misc import compute_iou_mat, unpickle
from utils.vis import compute_ap, get_random_colors
from datasets import load_eval_datasets


def get_current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def get_match_res(matrix):
    """ find match with maximum cost.
    """
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return row_ind, col_ind


def get_ap_from_ret_item(ret_item):
    num_gt = len(ret_item["probe_gt"])
    rank_list = np.array([gitem["correct"] for gitem in ret_item["gallery"]])
    ap = compute_ap(rank_list, num_gt) if num_gt > 0 else 0.0
    return ap


class ItemCriterion:
    def __call__(self, ret_item_a, ret_item_b, *args, **kwargs):
        raise NotImplementedError()


class SingleItemCriterion(ItemCriterion):
    def __call__(self, ret_item_a, ret_item_b, *args, **kwargs):
        raise NotImplementedError()


class DoubleItemCriterion(ItemCriterion):
    def __call__(self, ret_item_a, ret_item_b, *args, **kwargs):
        raise NotImplementedError()


class Top1MissedCriterion(SingleItemCriterion):
    def __call__(self, ret_item_a, ret_item_b, *args, **kwargs):
        return ret_item_a["gallery"][0]["correct"] != 1


class Top1BetterCriterion(DoubleItemCriterion):
    def __call__(self, ret_item_a, ret_item_b, *args, **kwargs):
        return ret_item_a["gallery"][0]["correct"] > ret_item_b["gallery"][0]["correct"]


class APBetterCriterion(DoubleItemCriterion):
    def __call__(self, ret_item_a, ret_item_b, *args, **kwargs):
        return ret_item_a["probe_ap"] > ret_item_b["probe_ap"]


class APMissedCriterion(SingleItemCriterion):
    def __call__(self, ret_item_a, ret_item_b, *args, **kwargs):
        return ret_item_a["probe_ap"] != 1.0


class PickleResDiffer:
    def __init__(self,
                 pickle_path1,
                 pickle_path2,
                 pickle1=None,
                 pickle2=None,
                 focus_a=True,  # set to True to focus results of pickle1.
                 ):
        self.ppath1 = pickle_path1
        self.ppath2 = pickle_path2
        self.focus_a = focus_a

        if pickle1 is None:
            self.pickle1 = unpickle(self.ppath1)
        else:
            self.pickle1 = pickle1

        if pickle2 is None:
            self.pickle2 = unpickle(self.ppath2)
        else:
            self.pickle2 = pickle2
        self.change_focus(self.focus_a)

    def change_focus(self, focus_a=True):
        self.focus_a = focus_a
        self.focus_pickle = self.pickle1 if self.focus_a else self.pickle2
        self.comp_pickle = self.pickle2 if self.focus_a else self.pickle1
        self._load_other_params()

    def _load_other_params(self):
        eval_args = self.focus_pickle["eval_args"]
        dataset = load_eval_datasets(eval_args)
        self.eval_args = eval_args
        self.dataset = dataset

        # mapping
        self.gallery_name_to_idx_map = dict([
            (item["im_name"], idx)
            for idx, item in enumerate(self.dataset.roidb)
        ])
        # hard to deal with duplicate probe.
        self.query_name_to_idx_map = dict([
            (item["im_name"], idx)
            for idx, item in enumerate(self.dataset.probes)
        ])

    def get_target_items(self, criterion: DoubleItemCriterion):
        focus_res = self.focus_pickle["eval_rank"]["results"]
        comp_res = self.comp_pickle["eval_rank"]["results"]

        assert len(focus_res) == len(comp_res)

        target_indices = []
        comp_indices = []
        for qidx, (ret_a, ret_b) in enumerate(zip(focus_res, comp_res)):
            if not criterion(ret_a, ret_b):
                continue

            qbox = np.asarray(ret_a["probe_roi"]).flatten()
            top1_box = np.asarray(ret_a["gallery"][0]["roi"]).flatten()
            top1_img = ret_a["gallery"][0]["img"]
            gidx = self.gallery_name_to_idx_map[top1_img]

            target_indices.append([qidx, qbox, gidx, top1_box])

            comp_qbox = np.asarray(ret_b["probe_roi"]).flatten()
            comp_box = np.asarray(ret_b["gallery"][0]["roi"]).flatten()
            comp_top1_img = ret_b["gallery"][0]["img"]
            comp_gidx = self.gallery_name_to_idx_map[comp_top1_img]
            comp_indices.append([qidx, comp_qbox, comp_gidx, comp_box])
        return target_indices, comp_indices

    def get_target_single_items(self, criterion: SingleItemCriterion):
        focus_res = self.focus_pickle["eval_rank"]["results"]
        target_indices = []
        for qidx, ret_a in enumerate(focus_res):
            if not criterion(ret_a, None):
                continue

            qbox = np.asarray(ret_a["probe_roi"]).flatten()
            top1_box = np.asarray(ret_a["gallery"][0]["roi"]).flatten()
            top1_img = ret_a["gallery"][0]["img"]
            gidx = self.gallery_name_to_idx_map[top1_img]

            target_indices.append(qidx, qbox, gidx, top1_box)
        return target_indices


class PickleVisualizer:
    def __init__(self, pkl_path, pkl=None):
        self.pkl_path = pkl_path
        if pkl is None:
            self.pkl = unpickle(self.pkl_path)
        else:
            self.pkl = pkl
        root = self.pkl_path + ".vis_pickle"
        self.base_root = os.path.join(root, get_current_time())
        self._load_other_params()
        self._register_color()

    def _load_other_params(self):
        eval_args = self.pkl["eval_args"]
        dataset = load_eval_datasets(eval_args)
        self.eval_args = eval_args
        self.dataset = dataset

        # mapping
        self.gallery_name_to_idx_map = dict([
            (item["im_name"], idx)
            for idx, item in enumerate(self.dataset.roidb)
        ])
        # hard to deal with duplicate probe.
        self.query_name_to_idx_map = dict([
            (item["im_name"], idx)
            for idx, item in enumerate(self.dataset.probes)
        ])

    def find_target_ind_by_boxes(self, target_box, boxes):
        """ target_box and boxes are all in np.ndarray
        """
        target_box = target_box.flatten()[:4].reshape(1, 4)
        iou = compute_iou_mat(target_box, boxes).flatten()
        ind = iou.argmax()

        try:  # TODO: is this necessary?
            assert iou[ind] > 0.8, "Not Match!?"
        except Exception as ex:
            print("Boxes not matched!")
            from IPython import embed
            embed()
            raise(ex)
        return ind

    def find_target_ind_by_features(self, target_feat, features):
        assert features.ndim == 2
        assert len(target_feat) == features.shape[1]

        target_feat = target_feat.flatten().reshape(1, -1)
        sim_matrix = np.matmul(target_feat, features.T).flatten()
        ind = sim_matrix.argmax()

        try:  # TODO: is this necessary?
            assert sim_matrix[ind] > 0.8, "Not Match!?"
        except Exception as ex:
            print("Boxes not matched!")
            from IPython import embed
            embed()
            raise(ex)
        return ind

    def write_signature(self):
        vfile = os.path.join(self.base_root, "version.txt")
        with open(vfile, "w") as f:
            f.write(get_current_time() + "\n")
            f.write(self.pkl_path + "\n")

    def _register_color(self):
        # in BGR mode
        self.colors = np.random.randint(0, 256, size=(100, 3), dtype=np.int).tolist()
        self.query_color = (255, 0, 0)  # BLUE
        self.gallery_match_color = (0, 255, 0)  # GREEN
        self.gallery_miss_color = (0, 0, 255)  # RED
        self.normal_color = (255, 255, 255)  # black

    def visualize_cmm_search(
                    self,
                    query_inds: list,
                    gallery_inds: list,
                    query_boxes: list,
                    gallery_boxes: list,
                    matched_flag=True,
                    ):

        probes = self.dataset.probes
        gallery = self.dataset.roidb

        for qind, gind, qbox, gbox in \
                zip(query_inds, gallery_inds, query_boxes, gallery_boxes):

            qitem, gitem = probes[qind], gallery[gind]
            qim_name, gim_name = qitem["im_name"], gitem["im_name"]
            save_img_root = os.path.join(self.base_root, f"{qim_name}_{gim_name}")
            os.makedirs(save_img_root, exist_ok=True)

            qfeats = self.pkl["query_features"][qind]
            gfeats = self.pkl["gallery_features"][gind]
            qboxes = self.pkl["query_boxes"][qind][:, :4]
            gboxes = self.pkl["gallery_boxes"][gind][:, :4]

            target_qind = self.find_target_ind_by_boxes(qbox, qboxes)
            target_gind = self.find_target_ind_by_boxes(gbox, gboxes)

            sim_mat = np.matmul(qfeats, gfeats.T)
            mridx, mcidx = get_match_res(sim_mat)

            # match label
            qm_labels = np.zeros(len(qfeats))
            gm_labels = np.zeros(len(gfeats))
            scores = sim_mat[mridx, mcidx]
            for i, (mx, my) in enumerate(zip(mridx, mcidx)):
                qm_labels[mx] = i + 1
                gm_labels[my] = i + 1

            # query
            img = cv2.imread(qitem["path"])
            dimg = img.copy()
            for i, box in enumerate(qboxes):
                x1, y1, x2, y2 = [int(x) for x in box.flatten()[:4]]
                mlabel = int(qm_labels[i])
                if i == target_qind:
                    color = self.query_color
                    thickness = 2
                else:
                    color = get_random_colors(mlabel)
                    thickness = 1
                cv2.rectangle(dimg, (x1, y1), (x2, y2), color, thickness=thickness)
            save_img_name = f"query_{qim_name}"
            save_img_name = os.path.join(save_img_root, save_img_name)
            cv2.imwrite(save_img_name, dimg)

            # gallery
            img = cv2.imread(gitem["path"])
            dimg = img.copy()
            for i, box in enumerate(gboxes):
                x1, y1, x2, y2 = [int(x) for x in box.flatten()[:4]]
                mlabel = int(gm_labels[i])
                if i == target_gind:
                    color = self.gallery_match_color if matched_flag else self.gallery_miss_color
                    thickness = 2
                else:
                    color = get_random_colors(mlabel)
                    thickness = 1
                cv2.rectangle(dimg, (x1, y1), (x2, y2), color, thickness=thickness)
            save_img_name = f"gallery_{gim_name}"
            save_img_name = os.path.join(save_img_root, save_img_name)
            cv2.imwrite(save_img_name, dimg)
        self.write_signature()

    def visualize_search(
                    self,
                    query_inds: list,
                    gallery_inds: list,
                    query_boxes: list,
                    gallery_boxes: list,
                    matched_flag=True,
                   ):

        probes = self.dataset.probes
        gallery = self.dataset.roidb

        for qind, gind, qbox, gbox in \
                zip(query_inds, gallery_inds, query_boxes, gallery_boxes):

            qitem, gitem = probes[qind], gallery[gind]
            qim_name, gim_name = qitem["im_name"], gitem["im_name"]
            save_img_root = os.path.join(self.base_root, f"{qim_name}_{gim_name}")
            os.makedirs(save_img_root, exist_ok=True)

            qboxes = self.pkl["query_boxes"][qind][:, :4]
            gboxes = self.pkl["gallery_boxes"][gind][:, :4]

            target_qind = self.find_target_ind_by_boxes(qbox, qboxes)
            target_gind = self.find_target_ind_by_boxes(gbox, gboxes)

            # query
            img = cv2.imread(qitem["path"])
            dimg = img.copy()
            x1, y1, x2, y2 = [int(x) for x in qboxes[target_qind].flatten()[:4]]
            cv2.rectangle(dimg, (x1, y1), (x2, y2), self.query_color, thickness=2)
            save_img_name = f"query_{qim_name}"
            save_img_name = os.path.join(save_img_root, save_img_name)
            cv2.imwrite(save_img_name, dimg)

            # gallery
            img = cv2.imread(gitem["path"])
            dimg = img.copy()
            x1, y1, x2, y2 = [int(x) for x in gboxes[target_gind].flatten()[:4]]
            color = self.gallery_match_color if matched_flag else self.gallery_miss_color
            cv2.rectangle(dimg, (x1, y1), (x2, y2), color, thickness=2)
            save_img_name = f"gallery_{gim_name}"
            save_img_name = os.path.join(save_img_root, save_img_name)
            cv2.imwrite(save_img_name, dimg)
        self.write_signature()


def test():
    baseline_pkl_path = "exps/exps_cuhk/checkpoint.pth.eval.pkl"
    cmm_pkl_path = "exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl"
    differ = PickleResDiffer(cmm_pkl_path, baseline_pkl_path)
    criterion = Top1BetterCriterion()

    target_indices, comp_indices = \
        differ.get_target_items(criterion)
    qinds, qboxes, ginds, gboxes = zip(*target_indices)

    visualizer = PickleVisualizer(cmm_pkl_path, pkl=differ.focus_pickle)
    visualizer.visualize_cmm_search(qinds, ginds, qboxes, gboxes)

    qinds, qboxes, ginds, gboxes = zip(*comp_indices)
    visualizer = PickleVisualizer(baseline_pkl_path, pkl=differ.comp_pickle)
    visualizer.visualize_search(qinds, ginds, qboxes, gboxes, False)

    from IPython import embed
    embed()


def test_acae():
    import sys
    start_inds = int(sys.argv[1])
    print(start_inds)

    from evaluation.vis_acae import CTXFeatureMapVisualizer, CTXAttnWeightsVisualizer
    baseline_pkl_path = "exps/exps_cuhk/checkpoint.pth.eval.pkl"
    acae_pkl_path = "exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl"

    differ = PickleResDiffer(acae_pkl_path, baseline_pkl_path)
    criterion = Top1BetterCriterion()
    target_indices, comp_indices = \
        differ.get_target_items(criterion)
    del differ

    qinds, qboxes, ginds, gboxes = zip(*target_indices)
    visualizer = CTXFeatureMapVisualizer(os.path.dirname(acae_pkl_path))
    # visualizer.vis_feature_map(qinds, ginds, qboxes, gboxes)
    # start_inds = 80
    step = 20
    for idx, (qind, gind, qbox, gbox) in tqdm(enumerate(zip(qinds, ginds, qboxes, gboxes))):
        if idx < start_inds or idx >= start_inds + step:
            continue
        print(idx)
        visualizer.vis_feature_map([qind], [gind], [qbox], [gbox])

    # visualizer = CTXAttnWeightsVisualizer(os.path.dirname(acae_pkl_path))
    # visualizer.draw_attn_weights(qinds, ginds, qboxes, gboxes)

    # visualize results in baseline
    # qinds, qboxes, ginds, gboxes = zip(*comp_indices)
    # visualizer = PickleVisualizer(baseline_pkl_path, pkl=differ.comp_pickle)
    # visualizer.visualize_search(qinds, ginds, qboxes, gboxes, False)


if __name__ == '__main__':
    test_acae()
