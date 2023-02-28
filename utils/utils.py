"""
Miscellaneous utilities
"""
import torch
import logging

from math import ceil


def get_normal_stats(tensor):
    return tensor.mean(), tensor.std()


class TrainingProgressTracker(object):
    stats = ['loss', 'acc', 'time', 'lr']

    def __init__(self, start_epoch, train_size, val_size, freq):
        self.train_size = train_size
        self.val_size = val_size
        self.freq = freq
        self.epoch = start_epoch

        self.progress = 0

        self.init_stats()

    @property
    def should_write(self):
        return (
            self.progress != 0 and not (self.progress + 1) % self.freq
            or (self.progress + 1 == self.train_size)
        )

    @property
    def scaling_factor(self):
        if self.progress + 1 > self.train_size - self.train_size % self.freq:
            return self.train_size % self.freq
        return self.freq

    def init_stats(self):
        for stat in TrainingProgressTracker.stats:
            setattr(self, stat, 0.)

    def damp_progress(self, **kwargs):
        for stat in TrainingProgressTracker.stats:
            try:
                new_stat = getattr(self, stat) + kwargs[stat] / self.scaling_factor
                setattr(self, stat, new_stat)
            except:
                raise KeyError(f'Tracking of training statistic {stat} is not implemented '\
                               f'the list of allowed statistics {TrainingProgressTracker.stats}')

    def write_progress(self):
        logging.info(f'Epoch [{self.epoch}] [{self.progress+1}/{self.train_size}]:    ' +
                     f'Loss: {self.loss:.6f}    ' +
                     f'Top1: {self.acc:.6f}    ' +
                     f'Time: {self.time:.6f}    ' +
                     f'LR: {self.lr:.6f}')

        self.init_stats()

    def step(self, **kwargs):
        self.damp_progress(**kwargs)
        if self.should_write:
            self.write_progress()

        self.progress += 1
        if self.progress > self.train_size - 1:
            self.progress = 0
            self.epoch += 1

    def val_info(self, epoch_num, val_loss, val_correct):

        logging.info(f'Epoch [{epoch_num}] Test set: Average loss: {val_loss:.4f}, ' +
                     f'Top1: {val_correct:.0f}/{self.val_size} ' +
                     f'({100. * val_correct / self.val_size:.2f}%)\n'
                     )

    @staticmethod
    def _key_in_metas(key, metas):
        return any([key in meta for meta in metas])


def preprocess_for_device(args):
    if torch.cuda.is_available() and not args.cpu:
        args.device = torch.device('cuda')
        if args.gpus is not None:
            try:
                args.gpus = list(map(int, args.gpus.split(',')))
            except:
                raise ValueError('GPU_ERROR: argument --gpus should be a comma-separated list of integers')
            num_gpus = torch.cuda.device_count()
            if any([gpu_id >= num_gpus for gpu_id in args.gpus]):
                raise ValueError('GPU_ERROR: specified gpu ids are not valid, check that gpu_id <= num_gpus')
            torch.cuda.set_device(args.gpus[0])
    else:
        args.gpus = -1
        args.device = torch.device('cpu')
    return args


def percentile(tensor, p):
    """
    Returns percentile of tensor elements
    Arguments:
        tensor {torch.Tensor} -- a tensor to compute percentile
        p {float} -- percentile (values in [0,1])
    """
    if p > 1.:
        raise ValueError(f'Percentile parameter p expected to be in [0, 1], found {p:.5f}')
    k = ceil(tensor.numel() * (1 - p))
    if p == 0:
        return -1  # by convention all param_stats >= 0
    return torch.topk(tensor.view(-1), k)[0][-1]

