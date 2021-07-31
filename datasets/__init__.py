import sys
from .cuhk_sysu import CUHK_SYSU
from .prw import PRW

from datasets.transforms import get_transform

sys.path.append("./")


def build_train_cuhk(root):
    transform = get_transform(True)
    imdb = CUHK_SYSU(root, transform, "train")
    return imdb


def build_dataloader(args):
    pass


def load_eval_datasets(args):
    if args.dataset_file == "cuhksysu":
        root = "data/cuhk-sysu/"
        # transform would not be used, set to None for simplicity.
        imdb = CUHK_SYSU(root, transforms=None, mode="test")
    elif args.dataset_file == "prw":
        root = "data/prw"
        imdb = PRW(root, transforms=None, mode="test")
    else:
        raise NotImplementedError(f"dataset {args.datset_file} no found.")
    return imdb
