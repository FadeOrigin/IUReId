from __future__ import print_function, absolute_import
import sys;
import time
from collections import OrderedDict
import pdb
import numpy;
import csv;
import os;

import torch
import numpy as np
import shutil;

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

from torch.autograd import Variable
import reid.evaluation_metrics.classification;
from .utils import to_torch
from .utils import to_numpy
import pdb


def extract_cnn_feature(model, inputs, output_feature=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=1, output_feature=None):
    model.eval()
    #model.train();
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, output_feature)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    return cmc_scores['market1501'][0]


def reranking(query_features, gallery_features, query=None, gallery=None, k1=20, k2=6, lamda_value=0.3):
        x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        feat = torch.cat((x, y))
        query_num, all_num = x.size(0), feat.size(0)
        feat = feat.view(all_num, -1)

        dist = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        dist = dist + dist.t()
        dist.addmm_(1, -2, feat, feat.t())

        original_dist = dist.numpy()
        all_num = original_dist.shape[0]
        original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
        V = np.zeros_like(original_dist).astype(np.float16)
        initial_rank = np.argsort(original_dist).astype(np.int32)

        print('starting re_ranking')
        for i in range(all_num):
            # k-reciprocal neighbors
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(k1 / 2)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
        original_dist = original_dist[:query_num, ]
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float16)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

        final_dist = jaccard_dist * (1 - lamda_value) + original_dist * lamda_value
        del original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:query_num, query_num:]
        return final_dist


def galleryTest(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self,gallery_loader, gallery, output_feature=None):
        gallery_features, labels = extract_features(self.model, gallery_loader, 1, output_feature);
        features = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0);
        groundTruthLabel=torch.cat([labels[f].unsqueeze(0) for f, _, _ in gallery], 0);
        prediction,accuracy = reid.evaluation_metrics.classification.getPredictionAndAccuracy(features, groundTruthLabel);

        # write to csv
        fileNames=[f for f, _, _ in gallery];
        predictionNumpy=numpy.array(prediction.cpu()).astype(int);
        groundTruthNumpy=numpy.array(groundTruthLabel);
        with open('IlluminationPrediction.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            for fileIndex in range(len(fileNames)):
                writer.writerow([fileNames[fileIndex],str(predictionNumpy[fileIndex]),str(groundTruthNumpy[fileIndex])]);

        #numpy.savetxt('label_prediction_numpy.txt', numpy.array(prediction.cpu()));
        #numpy.savetxt('label_ground_truth_numpy.txt', numpy.array(groundTruthLabel));

        print(accuracy);

    def generateIlluminationLabel(self,data_dir,dataset,dataVersion,subDirectory,gallery_loader, gallery, output_feature=None):

        labelDictionary={1:30,2:40,3:50,4:60,5:80,6:100,7:120,8:150,9:180,10:210,11:250,12:290,13:330};
        invertLabelDictionary={30:1,40:2,50:3,60:4,80:5,100:6,120:7,150:8,180:9,210:10,250:11,290:12,330:13};

        gallery_features, labels = extract_features(self.model, gallery_loader, 1, output_feature);
        features = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0);
        groundTruthLabel=torch.cat([labels[f].unsqueeze(0) for f, _, _ in gallery], 0);
        prediction,_ = reid.evaluation_metrics.classification.getPredictionAndAccuracy(features, groundTruthLabel);

        fileNames=[f for f, _, _ in gallery];
        predictionNumpy=numpy.array(prediction.cpu()).astype(int);
        inputDataDirectory = data_dir + "/" + dataset + "/" + dataVersion + "/"+subDirectory;
        outputDataDirectory=data_dir+"/"+dataset+"/"+dataVersion+"/"+subDirectory+"_output";
        if os.path.exists(outputDataDirectory):
            shutil.rmtree(outputDataDirectory);
        os.makedirs(outputDataDirectory);

        eq0Sum=0;
        eq1Sum=0;
        eq2Sum=0;
        gt2Sum=0;
        gt5Sum=0;
        errorSum=0;

        for fileIndex in range(len(fileNames)):
            inputImageFileName=fileNames[fileIndex];
            relabeledPredictionLabel=predictionNumpy[fileIndex];

            inputImageFileNameInformation = inputImageFileName.split("_");

            predictionLabel=labelDictionary[relabeledPredictionLabel];
            groundTruthLabel=int(inputImageFileNameInformation[2][1:]);
            relabeledGroundTruthLabel=invertLabelDictionary[groundTruthLabel];
            error=abs(relabeledPredictionLabel-relabeledGroundTruthLabel);
            errorSum=errorSum+error;
            if error==0:
                eq0Sum=eq0Sum+1;
            if error==1:
                eq1Sum=eq1Sum+1;
            if error==2:
                eq2Sum=eq2Sum+1;
            if error>2:
                gt2Sum=gt2Sum+1;
            if error>5:
                gt5Sum=gt5Sum+1;

            inputImageFileNameInformation.insert(3,"eg"+str(predictionLabel));
            outputImageFileName = "_".join(inputImageFileNameInformation);
            shutil.copyfile(inputDataDirectory + "/" + inputImageFileName,
                            outputDataDirectory + "/" + outputImageFileName);

        print(errorSum/len(fileNames));
        print(eq0Sum / len(fileNames));
        print(eq1Sum / len(fileNames));
        print(eq2Sum / len(fileNames));
        print(gt2Sum / len(fileNames));
        print(gt5Sum / len(fileNames));





