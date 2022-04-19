import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pathlib import Path
import os
import logging
import sys
import time
import shutil

import torch
import torch.nn as nn
import torch.utils



# ---------- UnNormalize utils ----------
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# ---------- find outliers ----------
def detect_outliers(data, threshold=3):
    Q1 = np.quantile(data, 0.25, axis=0)
    Q3 = np.quantile(data, 0.75, axis=0)
    IQR = Q3 - Q1

    outliers = []

    for (i,point) in enumerate(data):
        if data[i][0] < Q3[0]+1.5*IQR[0] and data[i][0]> Q1[0]-1.5*IQR[0]:
            if data[i][1] < Q3[1]+1.5*IQR[1] and data[i][1]> Q1[1]-1.5*IQR[1]:
                pass
            else:
                outliers.append(i)
        else:
            outliers.append(i)
            
    return outliers


# ---------- plot t-SNE ----------
ColorMap = ["#949483", "#f47b7b", "#9f1f5c", "#ef9020", "#00af3e", "#85b7e2", "#29245c", "#ffd616", "#e5352b", "#e990ab", "#0081b4", "#96cbb3", "#91be3e", "#39a6dd", "#eb0973", "#333c41"]
def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    ids = list(range(1, np.shape(data)[0]+1))
    # print(label)

    sns.set_context("paper")
    fig = plt.figure()
    for i in range(data.shape[0]):
        if (label[i]==-1):
            cluster_color = (0.6, 0.6, 0.6)
        else:
            cluster_color = ColorMap[label[i]%16]
        plt.text(data[i, 0], data[i, 1], str(ids[i]),color=cluster_color, fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    return fig

'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)

        config_dir = self.job_dir / 'config.txt'
        #if not os.path.exists(config_dir):
        if args.resume:
            with open(config_dir, 'a') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

#label smooth
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss