# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import numpy as np

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        self.writer.add_images(tag, images, step)

    def histo_summary(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def flush(self):
        self.writer.flush()