from logging import Logger
from torch.utils.tensorboard import SummaryWriter
import os

import torch
from collections import defaultdict, deque
import huepy as hue


class LogWriter(SummaryWriter):
    def __init__(self, root):
        os.makedirs(root, exist_ok=True)
        super().__init__(log_dir=root)
        self.epoch = 0
        self.iter = 0
        self.epoch_interval = 1
        self.iter_interval = 5
        self.data = dict()
        self.window_size = 10

    def update_item(self, key, val=None):
        if key not in self.data.keys():
            self.data[key] = []
        elif val is not None:
            self.data[key].append(val)

    def update_epoch(self, val=None):
        if val is None:
            self.epoch += 1
        else:
            assert isinstance(val, int)
            self.epoch = val

    def update_iter(self, val=None):
        if val is None:
            self.iter += 1
        else:
            assert isinstance(val, int)
            self.iter = val

    def _write_to_file(self, data, file):
        with open(os.path.join(self.log_dir, file)) as f:
            f.write(data)

    def save_dict_iter(self, data: dict):
        """ save data in each iteration interval
        """
        if self.iter % self.iter_interval == 0:
            for key, val in data.items():
                self.update_item(key, val)
            self.add_scalars("iter", data, global_step=self.iter)
        self.update_iter()

    def save_dict_epoch(self, data: dict):
        """ save data in each epoch interval
        """
        if self.epoch % self.epoch_interval == 0:
            self.add_scalars("epoch", data, global_step=self.epoch)
        self.update_epoch()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(SummaryWriter):
    def __init__(self, root, delimiter="\t", /, *args, **kwargs):
        super().__init__(log_dir=root)
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def print_log(self, epoch, step, iters_per_epoch):
        print(hue.lightgreen('[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e'
                             % (epoch, step, iters_per_epoch,
                                self.meters['loss_value'].avg, self.meters['lr'].value)))
        if 'num_fg' in self.meters:
            print('\tfg/bg: %d/%d, time cost: %.4f' %
                      (self.meters['num_fg'].avg, 
                       self.meters['num_bg'].avg, 
                       self.meters['batch_time'].avg))
        else:
            print('\ttime cost: %.4f' % (self.meters['batch_time'].avg))
        print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_box: %.4f'
                  % (self.meters['loss_objectness'].avg, 
                     self.meters['loss_rpn_box_reg'].avg,
                     self.meters['loss_box_reg'].avg))
        print('\tdet_cls: %.4f, reid_cls: %.4f'
                  % (self.meters['loss_detection'].avg, 
                     self.meters['loss_reid'].avg))
