from __future__ import print_function, absolute_import
import time
import sys;

import torch
import torch.nn.functional;
import reid.utils;
from torch.autograd import Variable
import reid.loss.label_smooth_cross_entropy;

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        losses = AverageMeter()
        accuracy_eq0=AverageMeter()
        accuracy_eq1 = AverageMeter()
        accuracy_eq2 = AverageMeter()
        accuracy_le5 = AverageMeter()
        accuracy_le10 = AverageMeter()

        for i, inputs in enumerate(data_loader):

            inputs, targets = self._parse_data(inputs)

            loss, accuracy = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))

            accuracy_eq0.update(accuracy[0], targets.size(0))
            accuracy_eq1.update(accuracy[1], targets.size(0))
            accuracy_eq2.update(accuracy[2], targets.size(0))
            accuracy_le5.update(accuracy[3], targets.size(0))
            accuracy_le10.update(accuracy[4], targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Precision_Eq0 {:.2%} ({:.2%})\t'
                      'Precision_Eq1 {:.2%} ({:.2%})\t'
                      'Precision_Eq2 {:.2%} ({:.2%})\t'
                      'Precision_Le5 {:.2%} ({:.2%})\t'
                      'Precision_Le10 {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              losses.val, losses.avg,
                              accuracy_eq0.val, accuracy_eq0.avg,
                              accuracy_eq1.val, accuracy_eq1.avg,
                              accuracy_eq2.val, accuracy_eq2.avg,
                              accuracy_le5.val, accuracy_le5.avg,
                              accuracy_le10.val, accuracy_le10.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, fnames, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            #print(targets.data);
            #print(outputs.data.shape);
            loss = self.criterion(outputs, targets)
            prediction, accuracy = reid.evaluation_metrics.classification.getPredictionAndAccuracy(outputs.data, targets.data+1.0);
            prec = accuracy;
        elif isinstance(self.criterion, reid.loss.label_smooth_cross_entropy.CrossEntropyLabelSmooth):
            loss = self.criterion(outputs, targets)
            prediction, accuracy = reid.evaluation_metrics.classification.getPredictionAndAccuracy(outputs.data,targets.data + 3.0);
            prec = accuracy;
        elif isinstance(self.criterion,torch.nn.MSELoss):
            #prediction,prec=reid.evaluation_metrics.classification.getPredictionAndAccuracy(outputs.data, targets.data);
            #prediction=prediction[0].float();
            #target = targets.float();
            #prediction.requires_grad = True;

            one_hot = torch.zeros(outputs.shape[0], outputs.shape[1]).cuda().scatter_(1, targets.view(outputs.shape[0],1), 1);
            #loss=self.criterion(torch.nn.functional.softmax(outputs, 1),one_hot);
            loss = self.criterion(outputs, one_hot);

            #print(outputs);
            #print(outputs.shape);
            #target=targets.float().view(outputs.shape[0],1);
            #print(target);
            #print(target.shape);
            #sys.exit();
            #print(targets.float().view(outputs.shape[0],1));
            #loss = self.criterion(outputs, target);
            prediction, accuracy = reid.evaluation_metrics.classification.getPredictionAndAccuracy(outputs.data, targets.data+3.0);
            prec = accuracy;
        elif isinstance(self.criterion, TripletLoss):
            loss, _ = self.criterion(outputs, targets)
            prediction, accuracy = reid.evaluation_metrics.classification.getPredictionAndAccuracy(outputs.data, targets.data+3.0);
            prec = accuracy;
        else:
            print(self.criterion);
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class CamStyleTrainer(object):
    def __init__(self, model, criterion, camstyle_loader):
        super(CamStyleTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.camstyle_loader = camstyle_loader
        self.camstyle_loader_iter = iter(self.camstyle_loader)

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            try:
                camstyle_inputs = next(self.camstyle_loader_iter)
            except:
                self.camstyle_loader_iter = iter(self.camstyle_loader)
                camstyle_inputs = next(self.camstyle_loader_iter)
            inputs, targets = self._parse_data(inputs)
            camstyle_inputs, camstyle_targets = self._parse_data(camstyle_inputs)
            loss, prec1 = self._forward(inputs, targets, camstyle_inputs, camstyle_targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs.cuda())
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, camstyle_inputs, camstyle_targets):
        outputs = self.model(inputs)
        camstyle_outputs = self.model(camstyle_inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        camstyle_loss = self._lsr_loss(camstyle_outputs, camstyle_targets)
        loss += camstyle_loss
        return loss, prec

    def _lsr_loss(self, outputs, targets):
        num_class = outputs.size()[1]
        targets = self._class_to_one_hot(targets.data.cpu(), num_class)
        targets = Variable(targets.cuda())
        outputs = torch.nn.LogSoftmax()(outputs)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def _class_to_one_hot(self, targets, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, 0.9)
        targets_onehot.add_(0.1 / num_class)
        return targets_onehot

