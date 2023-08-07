__author__ = 'yuezhou'

import cv2
import numpy as np
import os

# torch dataset
from torch.utils.data import Dataset

# import external libs
from utils.data_util import numpy2tensor

# register
from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class TwoDataset(Dataset):
    def __init__(self, args, sanity_check=False):
        self.root = args['data_root']
        self.force_large = args['force_large']
        self.force_middle = args['force_middle']
        self.max_force = args['max_force']

    def __getitem__(self, index):
        zero = cv2.imread(os.path.join(self.root,'zero.png'))
        midd = cv2.imread(os.path.join(self.root,'midd.png'))
        defo = cv2.imread(os.path.join(self.root,'defo.png'))

        gt_flow = np.load(os.path.join(self.root, 'gt_flow_combined.npy'))
        gt_flow_zero2mid = np.load(os.path.join(self.root, 'gt_flow_mid2zero.npy'))
        gt_flow_mid2defo = np.load(os.path.join(self.root, 'gt_flow_defo2mid.npy'))

        self.height, self.width = gt_flow.shape[0], gt_flow.shape[1]

        force_large = np.ones(shape=(self.height, self.width, 1),
                        dtype=np.float32) * self.force_large
        force_middle = np.ones(shape=(self.height, self.width, 1),
                        dtype=np.float32) * self.force_middle
        

        # resize input image
        zero = cv2.resize(zero, (self.width, self.height), interpolation = cv2.INTER_AREA)
        defo = cv2.resize(defo, (self.width, self.height), interpolation = cv2.INTER_AREA)

        # set to tensor
        zero_tensor = numpy2tensor(zero, normalize=True)
        midd_tensor = numpy2tensor(midd, normalize=True)
        defo_tensor = numpy2tensor(defo, normalize=True)

        flow_tensor = numpy2tensor(gt_flow, normalize=False)
        flow_zero2mid_tensor =  numpy2tensor(gt_flow_zero2mid, normalize=False)
        flow_mid2defo_tensor = numpy2tensor(gt_flow_mid2defo, normalize=False)

        force_large_tensor = numpy2tensor(force_large, normalize=True, max_force=self.max_force)
        force_middle_tensor = numpy2tensor(force_middle, normalize=True, max_force=self.max_force)


        return {
            'zero': zero_tensor,
            'midd': midd_tensor,
            'defo': defo_tensor,
            'gt_flow': flow_tensor,
            'flow_zero2mid': flow_zero2mid_tensor,
            'flow_mid2defo': flow_mid2defo_tensor,
            'force_large': force_large_tensor,
            'force_middle': force_middle_tensor
        }

    def __len__(self):
        return 1