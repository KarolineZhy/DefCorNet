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
class SimDataset(Dataset):
    def __init__(self, args, sanity_check=False):
        self.root = args['data_root']
        self.sim_force = args['sim_force']
        self.max_force = args['max_force']
        self.width = args['img_width']
        self.height = args['img_height']


    def __getitem__(self, index):
        zero = cv2.imread(os.path.join(self.root,'zero.png'))
        defo = cv2.imread(os.path.join(self.root,'defo.png'))
        gt_flow = np.load(os.path.join(self.root, 'flow.npy'))
        
        force = np.ones(shape=(self.height, self.width, 1), dtype=np.float32) * self.sim_force

        # resize input image
        zero = cv2.resize(zero, (self.width, self.height), interpolation = cv2.INTER_AREA)
        defo = cv2.resize(defo, (self.width, self.height), interpolation = cv2.INTER_AREA)
        
        # set to tensor
        zero_tensor = numpy2tensor(zero, normalize=True)
        defo_tensor = numpy2tensor(defo, normalize=True)
        flow_tensor = numpy2tensor(gt_flow, normalize=False)
        force_tensor = numpy2tensor(force, normalize=True, max_force=self.max_force)

        return {
            'zero': zero_tensor,
            'defo': defo_tensor,
            'gt_flow': flow_tensor,
            'force': force_tensor
        }

    def __len__(self):
        return 1