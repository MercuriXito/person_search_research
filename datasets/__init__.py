import sys
from .cuhk_sysu import CUHK_SYSU
from .prw import PRW

from datasets.transforms import get_transform

sys.path.append("./")


def build_trainset(dataset_name, root, use_transform=True):
    transform = get_transform(use_transform)
    if dataset_name == "cuhk-sysu":
        imdb = CUHK_SYSU(root, transform, "train")
    elif dataset_name == "prw":
        imdb = PRW(root, transform, "train")
    return imdb


def load_eval_datasets(args):
    if args.dataset_file == "cuhk-sysu":
        root = "data/cuhk-sysu/"
        # transform would not be used, set to None for simplicity.
        imdb = CUHK_SYSU(root, transforms=None, mode="test")
    elif args.dataset_file == "prw":
        root = "data/prw"
        imdb = PRW(root, transforms=None, mode="test")
    else:
        raise NotImplementedError(f"dataset {args.dataset_file} no found.")
    return imdb
