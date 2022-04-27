import pickle as pkl
import json
import os
import random
from prettytable import PrettyTable
import numpy as np
import torch
import torch.distributed as dist

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def _fix_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def compute_iou_mat(a: np.ndarray, b: np.ndarray):
    m, n = len(a), len(b)

    a = np.expand_dims(a, 1)
    b = np.expand_dims(b, 0)
    a = a.repeat(n, 1).reshape(-1, a.shape[-1])
    b = b.repeat(m, 0).reshape(-1, a.shape[-1])

    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + \
        (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter

    mat = inter * 1.0 / union
    mat = mat.reshape(m, n)
    return mat


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


class CompareError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CompareTypeError(CompareError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CompareResError(CompareError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def compare(outs_a, outs_b, prefix=None):
    if prefix is None:
        prefix = []
    assert isinstance(prefix, list)
    try:
        if type(outs_a) != type(outs_b):
            raise CompareTypeError(f"{type(outs_a)} != {type(outs_b)}")
        res = True
        if isinstance(outs_a, dict):
            assert set(outs_a.keys()) == set(outs_b.keys())
            for k in outs_a.keys():
                va, vb = outs_a[k], outs_b[k]
                res = res and compare(va, vb, prefix=prefix+[f"{k}"])
        elif isinstance(outs_a, (list, tuple)):
            assert len(outs_a) == len(outs_b)
            for i in range(len(outs_a)):
                res = res and compare(outs_a[i], outs_b[i], prefix=prefix+[f"{i}"])
        elif isinstance(outs_a, torch.Tensor):
            res = res and (torch.sum(outs_a - outs_b != 0) == 0).item()
        elif isinstance(outs_a, np.ndarray):
            res = res and (np.sum(outs_a - outs_b != 0) == 0).item()
        elif isinstance(outs_a, (str, int, float)):
            res = res and (outs_a == outs_b)
        else:
            raise NotImplementedError(f"{type(outs_a)}")
        if not res:
            raise CompareResError()
    except Exception as ex:
        print("Exception occur in :{}".format(".".join(prefix)))
        raise(ex)
    return res
