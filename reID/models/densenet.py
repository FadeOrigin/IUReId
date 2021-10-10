from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb


__all__ = ['DenseNet', 'densenet121']


class DenseNet(nn.Module):
    __factory = {
        121: torchvision.models.densenet121,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(DenseNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) densenet
        if depth not in DenseNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = DenseNet.__factory[depth](pretrained=pretrained)
        self.base.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base.fc = nn.Sequential()

        if not self.cut_at_pooling:

            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.num_classes = num_classes

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

    def forward(self, x, output_feature=None):

        x = self.base.features(x)
        x = x.view(x.size(0), x.size(1))

        if self.norm:
            x = F.normalize(x)
        if self.dropout > 0:
            x = self.drop(x)
        x = self.classifier(x)
        return x

def densenet121(**kwargs):
    return DenseNet(121, **kwargs)

