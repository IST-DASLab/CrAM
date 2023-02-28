import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import warnings

__all__ = ['CosineLR']


class CosineLR(_LRScheduler):
    # Adapted from the cosine lr function here:
    # https://github.com/adityakusupati/STR/blob/master/utils/schedulers.py
    def __init__(self, optimizer, warmup_length, end_epoch, last_epoch=-1):
        self.warmup_length = warmup_length
        self.end_epoch = end_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def _warmup_lr(self):
        lrs = [base_lr * (self.last_epoch + 1) / self.warmup_length for base_lr in self.base_lrs]
        return lrs

    def _cosine_lr(self):
        e = self.last_epoch - self.warmup_length
        es = self.end_epoch - self.warmup_length
        lrs = [0.5 * (1 + np.cos(np.pi * e / es)) * base_lr for base_lr in self.base_lrs]
        return lrs

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch < self.warmup_length:
            updated_lrs = self._warmup_lr()
        else:
            updated_lrs = self._cosine_lr()
        return updated_lrs

