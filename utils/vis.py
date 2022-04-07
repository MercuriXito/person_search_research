import cv2
import torch
import numpy as np
import random
import time
import os


def check_save_path(path):
    dirname = os.path.dirname(path)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    return path


def compute_ap(ranks, num_gt=None):
    """ compute average precision for one ranking list.
    """
    if isinstance(ranks, list):
        ranks = np.asarray(ranks)
    if num_gt is None:
        num_gt = np.sum(ranks)
    else:
        assert isinstance(num_gt, int)
        assert num_gt >= np.sum(ranks)

    length = len(ranks)
    tps = np.cumsum(ranks)
    positives = np.arange(1, length+1)
    precisions = tps / positives
    recalls = np.zeros((length + 1, ))
    recalls[1:] = tps / num_gt

    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions)
    return ap


def cxcywh_to_xyxy(boxes: np.ndarray):
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    n_box_dim = boxes.shape[1]
    items = np.split(boxes, n_box_dim, axis=1)
    cx, cy, w, h = items[:4]
    x1 = cx - 0.5 * w
    x2 = cx + 0.5 * w
    y1 = cy - 0.5 * h
    y2 = cy + 0.5 * h
    return np.concatenate([x1, x2, y1, y2] + items[4:], axis=1)


def get_random_colors(ids=None):
    def _get_random_color(id=None):
        if id is None:
            id = random.randint(1, 1000)
        b = int(id * 997) % 255
        g = int(id * 4447) % 255
        r = int(id * 6563) % 255
        return (b, g, r)
    if not isinstance(ids, (tuple, list)):
        return _get_random_color(ids)
    else:
        return [_get_random_color(id) for id in ids]


def draw_boxes(image, boxes, save_path):
    boxes = cxcywh_to_xyxy(boxes)
    for box in boxes:
        x1, y1, x2, y2 = [int(x) for x in box[:4]]
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 0, 0), 3)
    save_path = check_save_path(save_path)
    cv2.imwrite(save_path, image)


def draw_boxes_text(
        image, boxes, str_texts=None, name="",
        write_output=False, normalize=False):

    if isinstance(image, torch.Tensor):
        assert image.ndim == 3
        assert image.size(0) == 3 or image.size(0) == 1
        cimage = image.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        cimage = image
    cimage = cimage.copy()
    if cimage.dtype != np.uint8:
        if normalize:
            image_mean = np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, -1)
            image_std = np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, -1)
            cimage = cimage * image_std + image_mean
        cimage = np.clip(cimage, 0, 1) * 255.0
        cimage = cimage.astype(np.uint8)
        # RGB2BGR
        cimage = cv2.cvtColor(cimage, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(boxes):
        text = str_texts[i] if str_texts is not None else ""
        x1, y1, x2, y2 = [int(x) for x in box.flatten()[:4]]
        cv2.rectangle(
            cimage, (x1, y1), (x2, y2),
            get_random_colors(i), 2
        )
        cv2.putText(cimage, text, (x1, y1), 0, 0.5, get_random_colors(i), 2)

    if write_output:
        if len(name) == 0:
            time_string = time.strftime("%H-%M-%S", time.localtime())
            cv2.imwrite("outputs/{}.png".format(time_string), cimage)
        else:
            cv2.imwrite(name, cimage)
    return cimage


def draw_heatmap(image, heatmap, alpha=0.6):
    """ return image * alpha + heatmap * (1 - alpha)
    """

    if image.dtype != np.uint8:
        # prcoess image
        image = np.clip(image, 0, 1) * 255.0
    if heatmap.dtype != np.uint8:
        # process heatmap
        heatmap = np.clip(heatmap, 0, 1) * 255.0
        heatmap = heatmap.astype(np.uint8)

    # heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = image * alpha + heatmap * (1-alpha)
    image = image.astype(np.uint8)
    return image


if __name__ == '__main__':
    pass
