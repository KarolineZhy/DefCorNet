__author__ = 'yuezhou'

# import basic lib
import cv2
import numpy as np
import os

from glob import glob
import pickle

# torch lib
from torch.utils.data import Dataset

# import external libs
from .transforms import apply_transform
from utils.data_util import numpy2tensor
from utils.flow_util import warp

from flow_vis import flow_to_color

from utils.registry import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class PrevDataset(Dataset):
    def __init__(self, args, sanity_check=True):

        self.data_root = args['data_root']
        self.aug_params = args.get('aug_params', None)

        self.mode = args['mode']
        if self.mode == 'train':
            self.raws = sorted(glob(os.path.join(self.data_root, '*', 'raw/*.png')))
            self.flows = sorted(glob(os.path.join(self.data_root, '*', 'flow/*.npy')))

        elif self.mode == 'test':
            self.raws = sorted(glob(os.path.join(self.data_root, args['test_dataset'], 'raw/*.png')))
            self.flows = sorted(glob(os.path.join(self.data_root, args['test_dataset'], 'flow/*.npy')))
        else:
            print('mode is wrong')


        self.sanity_check = sanity_check

        self.max_force = args['max_force']
        self.height = args['height']
        self.width = args['width']

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx):

        # get image path root by index
        raw_path = self.raws[idx]
        flow_path = self.flows[idx]

        # using opencv to load image
        raw = cv2.imread(raw_path)
        raw = cv2.resize(raw, (self.width, self.height), interpolation = cv2.INTER_AREA)
        flow = np.load(flow_path)
        mask = np.ones((256,256))

        # using raw path to get correspondent force and pose
        seq_dir = os.path.abspath(os.path.join(raw_path, "../.."))
        file_to_read = open(os.path.join(seq_dir, 'result_dict.pkl'), "rb")
        result_dict = pickle.load(file_to_read)

        raw_name = os.path.basename(raw_path)

        # here only use z direction force and pose
        force_value = result_dict[raw_name]['force']['z'] * -1
        force = np.ones(shape=(self.height, self.width, 1),
                        dtype=np.float32) * force_value  # force direction is nigated

        # we also need to load target by using the first image in the raw folder
        raw_dir = os.path.abspath(os.path.join(raw_path, ".."))
        target_file = os.path.join(raw_dir, 'raw_0000.png')
        target = cv2.imread(target_file)
        target = cv2.resize(target, (self.width, self.height), interpolation = cv2.INTER_AREA)
        
        # apply transformation
        raw, target, flow, force, mask = apply_transform(raw, target, flow, force, mask, self.aug_params)
        
        # convert numpy to tensor
        raw_tensor = numpy2tensor(raw, normalize=True)
        mask_tensor = numpy2tensor(mask, normalize=True)
        target_tensor = numpy2tensor(target, normalize=True)
        flow_tensor = numpy2tensor(flow, normalize=False)
        force_tensor = numpy2tensor(force, normalize=True, max_force=self.max_force)
        
    
        # sanity check
        if self.sanity_check:
            flow_vis = flow_to_color(flow)
            warped_tensor = warp(raw_tensor.unsqueeze(0), flow_tensor.unsqueeze(0))
            warped = warped_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

            concate = np.hstack((raw, target, (warped * 255).astype(np.uint8), flow_vis))
            cv2.putText(concate, 'force:{:4f}'.format(force_value), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 255, 255), 1)
            check_path = 'check/prev'
            image_path = os.path.join(check_path, 'index_{:04d}.png'.format(idx))
            cv2.imwrite(image_path, concate)

        return {'raw': raw_tensor,
                'mask': mask_tensor,
                'target': target_tensor,
                'flow': flow_tensor,
                'force': force_tensor
                }

