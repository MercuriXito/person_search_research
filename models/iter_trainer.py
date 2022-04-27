import torch
import time
import datetime
import os
import json
from typing import Iterable
from pathlib import Path

import numpy as np
import random
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

from models import build_models
from datasets import build_trainset
import utils.misc as utils
from utils.logger import MetricLogger
from utils.misc import ship_to_cuda, yaml_dump


def collate(batch):
    return list(zip(*batch))


def save_config(args, filename):
    from yacs.config import CfgNode
    from easydict import EasyDict
    if isinstance(args, CfgNode):
        with open(filename, "w") as f:
            f.write(args.dump())
    elif isinstance(args, EasyDict):
        # TODO: add support for easydict
        raise Exception("not used for EasyDict")
    else:
        # dict
        yaml_dump(args, filename)


# WarmuplrScheduler
# train in one iteration
class WarmupMultiStepLR(_LRScheduler):
    """ Chained lr scheduler, including Warmup strategy and multi-step decay.
    Call `step()` in each iteration. Original code from official fcos
    implementation.
    """
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def train_one_epoch(
        model, data_loader: Iterable, optimizer: torch.optim.Optimizer,
        lr_scheduler: _LRScheduler,
        device: torch.device, loss_weights: dict,
        epoch: int, state_dict: dict,
        max_norm: float = 0,
        ):

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5
    loader = iter(data_loader)

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        state_dict["iteration"] += 1
        images, targets = next(loader)
        images, targets = ship_to_cuda(images, targets, device)
        outputs, loss_dict = model(images, targets)
        losses = sum([loss_dict[k] * loss_weights[k] for k in loss_dict.keys() if k in loss_weights])

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(**loss_dict)
        metric_logger.update(loss=losses.item())
        metric_logger.update(grad_norm=grad_total_norm)
        torch.cuda.empty_cache()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    device = torch.device(args.device)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # build for dataset
    dataset = build_trainset(args.train.dataset, args.train.data_root)
    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=args.train.batch_size, drop_last=True)
    dataloader = DataLoader(
        dataset,
        num_workers=args.train.num_workers,
        batch_sampler=batch_sampler,
        pin_memory=True,
        collate_fn=collate,  # not enable collate here.
    )

    # build model
    model = build_models(args)
    model.to(device)

    # build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.train.lr_drop_epochs)

    # train iterations in one epoch on one single gpu.
    iteration_one_epoch = len(dataset) // args.train.batch_size
    milestones = [step * iteration_one_epoch for step in args.train.lr_drop_epochs]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.1,
        warmup_factor=1/3.0,
        warmup_iters=iteration_one_epoch,
    )

    # load previous trained model
    checkpoint_path = args.train.resume
    trained_epoch = 0
    if os.path.exists(checkpoint_path):
        params = torch.load(checkpoint_path, map_location="cpu")
        model_params = params["model"]
        missed, unexpected = model.load_state_dict(model_params, strict=False)
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")
        if len(missed) > 0:
            print(f"Missed keys: {missed}")

        # overwrite optimizer
        if "optimizer" in params and "lr_scheduler" in params and "epoch" in params:
            optimizer.load_state_dict(params["optimizer"])
            lr_scheduler.load_state_dict(params["lr_scheduler"])
            trained_epoch = params["epoch"]
    else:
        print(f"checkpoint: {checkpoint_path} does not exists.")
    start_epoch = trained_epoch + 1

    loss_weights = dict(**args.train.loss_weights)
    output_dir = Path(args.train.output_dir)
    save_config(args, output_dir / "config.yml")
    print("Start training")
    start_time = time.time()
    state_dict = dict(iteration=0)
    for epoch in range(start_epoch, args.train.epochs + 1):
        train_stats = train_one_epoch(
            model, dataloader, optimizer, lr_scheduler,
            device, loss_weights, epoch,
            state_dict, args.train.clip_max_norm
        )
        # lr_scheduler.step()
        if args.train.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if epoch % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.train.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_configs():
    from configs.faster_rcnn_default_configs import get_default_cfg
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    args = parser.parse_args()

    t_args = get_default_cfg()
    if len(args.cfg) > 0:
        if not os.path.exists(args.cfg):
            print(f"{args.cfg} not existed.")
        else:
            t_args.merge_from_file(args.cfg)
    t_args.freeze()
    print(t_args)
    if not os.path.exists(t_args.train.output_dir):
        os.makedirs(t_args.train.output_dir)
    return t_args


if __name__ == '__main__':
    args = get_configs()
    main(args)
