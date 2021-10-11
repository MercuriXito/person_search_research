import pickle as pkl
import json
import os
from prettytable import PrettyTable
import numpy as np
import torch
import torch.distributed as dist

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


# ==================================
# serialization and de-serialization
# ==================================
def check_save_path(path):
    dirname = os.path.dirname(path)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    return path


def pickle(data, file_path):
    file_path = check_save_path(file_path)
    with open(file_path, 'wb') as f:
        pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def json_load(path):
    with open(path, "rb") as f:
        data = json.load(f)
    return data


def json_dump(data, path):
    path = check_save_path(path)
    with open(path, "wb") as f:
        json.dump(data, f)


def yaml_dump(data, path):
    path = check_save_path(path)
    with open(path, "w") as f:
        dump(data, f, Dumper=Dumper)


def yaml_load(path):
    with open(path, "r") as f:
        data = load(f, Loader=Loader)
    return data


# ==================================
# distributed related
# ==================================
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# ==================================
# misc
# ==================================
def print_as_table(data, fieldnames=None):
    if isinstance(data, dict):
        fieldnames = list(data.keys())
        data = list(data.values())
    else:
        assert fieldnames is not None, \
            "fieldnames should not be none, if data is not dict."
    table = PrettyTable(field_names=fieldnames)
    table.add_row(data)
    print(table)
    return table.get_string()


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type) # not average
    return total_norm


def is_main_process():
    # TODO: multi-gpu not supported right now, so always return true.
    return True


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def ship_to_cuda(images, targets=None, device=None):
    if device is None:
        device = torch.device("cuda:0")
    images = [image.to(device) for image in images]
    if targets is not None:
        new_targets = []
        for target in targets:
            new_target = dict()
            for k, v in target.items():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    v = torch.as_tensor(v).to(device)
                new_target[k] = v
            new_targets.append(new_target)
        return images, new_targets
    else:
        return images
