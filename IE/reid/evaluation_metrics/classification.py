from __future__ import absolute_import

from ..utils import to_torch
import torch;
import numpy;
import sys;


def accuracy(output, target, topk=(1,)):
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret

def getPredictionAndAccuracy(output, groundTruthLabel, topk=(1,)):

    output, target = to_torch(output), to_torch(groundTruthLabel)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    label_prediction=pred[0]+1.0;
    label_diff = torch.abs(target - label_prediction);

    accuracy_eq0 = torch.sum(torch.eq(label_diff, 0).float()) / batch_size;
    accuracy_eq1 = torch.sum(torch.eq(label_diff, 1).float()) / batch_size;
    accuracy_eq2 = torch.sum(torch.eq(label_diff, 2).float()) / batch_size;
    accuracy_le5=torch.sum(torch.le(label_diff, 5).float()) / batch_size;
    accuracy_le10 = torch.sum(torch.le(label_diff, 10).float()) / batch_size;

    accuracy=torch.tensor([accuracy_eq0,accuracy_eq1,accuracy_eq2,accuracy_le5,accuracy_le10]);

    return label_prediction,accuracy;
