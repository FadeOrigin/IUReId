from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class Duke(object):

    def __init__(self, root,dataVersion):

        self.images_dir = osp.join(root,dataVersion)
        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.train, self.gallery= [],[];
        self.num_train_ids, self.num_gallery_ids= 0,0;
        self.load()

    def preprocess(self, path, relabel=True):
        #pattern = re.compile(r'([-\d]+)_c(\d)')
        #pattern = re.compile(r'([-\d]+)_([-\d]+)_c(\d)')
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            #pid, _,cam = map(int, pattern.search(fname).groups())
            pid, cam = map(int, pattern.search(fname).groups())
            #if pid == -1: continue
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

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path);
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False);

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  test  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))

