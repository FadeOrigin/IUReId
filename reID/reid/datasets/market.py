from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
import sys;


class Market(object):

    def __init__(self, root,dataVersion,cameraFilters):

        self.images_dir = osp.join(root,dataVersion);
        print(self.images_dir );
        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
        self.camstyle_path = 'reconstruct'
        self.train, self.query, self.gallery, self.camstyle = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_camstyle_ids = 0, 0, 0, 0
        self.load(cameraFilters)

    def preprocess(self, path,cameraFilters, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            for cameraFilter in cameraFilters:
                if fpath.find(cameraFilter)>-1:
                    fname = osp.basename(fpath)
                    pid, cam = map(int, pattern.search(fname).groups())
                    if pid == -1: continue
                    if relabel:
                        if pid not in all_pids:
                            all_pids[pid] = len(all_pids)
                    else:
                        if pid not in all_pids:
                            all_pids[pid] = pid
                    pid = all_pids[pid]
                    cam -= 1
                    ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self,cameraFilters):
        self.train, self.num_train_ids = self.preprocess(self.train_path,cameraFilters)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path,cameraFilters,False)
        self.query, self.num_query_ids = self.preprocess(self.query_path,cameraFilters,False)
        self.camstyle, self.num_camstyle_ids = self.preprocess(self.camstyle_path,cameraFilters)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
        print("  camstyle  | {:5d} | {:8d}"
              .format(self.num_camstyle_ids, len(self.camstyle)))
