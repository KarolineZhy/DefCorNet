__author__ = 'yuezhou'

# import basic lib
import cv2
import numpy as np
import os
from scipy.stats import norm

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

stiff_dict = {
    'human1': 1.58
}

mu = 1.5618638943451695 
sigma = 0.4944927602224852

@DATASET_REGISTRY.register()
class UltrasoundDataset(Dataset):
    def __init__(self, args, sanity_check=False):

        self.data_root = args['data_root']
        self.aug_params = args.get('aug_params', None)
        self.use_gaussian = args['use_gaussian']

        self.mode = args['mode']
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test' or self.mode == 'test_sweep':
            self.raws = sorted(glob(os.path.join(self.data_root, '*', 'raw/*.png')))
            self.flows = sorted(glob(os.path.join(self.data_root, '*', 'flow/*.npy')))
        else:
            print('mode is wrong')
        
        self.sanity_check = sanity_check

        self.max_force = args['max_force']
        self.img_height = args['img_height']
        self.img_width= args['img_width']


    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx):
        
        # get image path root by index
        raw_path = self.raws[idx]
        
        if len(self.flows) != 0:
            flow_path = self.flows[idx]
            flow = np.load(flow_path)
        else:
            flow = np.zeros((self.img_height, self.img_width, 2))

        # using opencv to load image
        raw = cv2.imread(raw_path)

        # using raw path to get correspondent force and pose
        seq_dir = os.path.abspath(os.path.join(raw_path, "../.."))
        file_to_read = open(os.path.join(seq_dir, 'result_dict.pkl'), "rb")
        result_dict = pickle.load(file_to_read)
        raw_name = os.path.basename(raw_path)
       

        # here only use z direction force and pose
        force_value = result_dict[raw_name]['force']['z'] * -1
        base_force = result_dict['raw_0000.png']['force']['z'] * -1
        
        pose_value = result_dict[raw_name]['pose']['position']['z']
        
        if self.mode == 'train' or self.mode == 'val' or self.mode=='test':
            stiff_value = result_dict[raw_name]['stiffness']
            force_value = force_value - base_force
        elif self.mode == 'test_sweep':
            stiff_value = stiff_dict['human1']
            force_value = force_value

        if self.use_gaussian:
            stiff_value = norm.pdf(stiff_value, mu, sigma)
        
        
        force = np.ones(shape=(self.img_height, self.img_width, 1),
                        dtype=np.float32) * force_value  # force direction is nigated
        pose = np.ones(shape=(self.img_height, self.img_width, 1), dtype=np.float32) * pose_value
        stiffness = np.ones(shape=(self.img_height, self.img_width, 1), dtype=np.float32) * stiff_value
        

        if self.mode == 'train' or self.mode == 'val' or self.mode=='test':
            # we also need to load target by using the first image in the raw folder
            raw_dir = os.path.abspath(os.path.join(raw_path, ".."))
            target_file = os.path.join(raw_dir, 'raw_0000.png')
            target = cv2.imread(target_file)

            # apply transformation
            if self.mode == 'train' or self.mode=='test':
                raw, target, flow, force, stiffness = apply_transform(raw, target, flow, force, stiffness, self.aug_params)
            
            # convert numpy to tensor
            raw_tensor = numpy2tensor(raw, normalize=True)
            target_tensor = numpy2tensor(target, normalize=True)
            flow_tensor = numpy2tensor(flow, normalize=False)
            force_tensor = numpy2tensor(force, normalize=True, max_force=self.max_force)
            pose_tensor = numpy2tensor(pose, normalize=False)
            stiffness_tensor = numpy2tensor(stiffness, normalize=False)
        
        elif self.mode == 'test_sweep':
            # we do not have target 
            raw_dir = os.path.abspath(os.path.join(raw_path, ".."))
            target = raw
            # use raw for target
            target_tensor = numpy2tensor(target, normalize=True)
            raw_tensor = numpy2tensor(raw, normalize=True)
            flow_tensor = numpy2tensor(flow, normalize=False)
            force_tensor = numpy2tensor(force, normalize=True, max_force=self.max_force)
            pose_tensor = numpy2tensor(pose, normalize=False)
            stiffness_tensor = numpy2tensor(stiffness, normalize=False)

        
            # sanity check
            if self.sanity_check:
                flow_vis = flow_to_color(flow)
                warped_tensor = warp(raw_tensor.unsqueeze(0), flow_tensor.unsqueeze(0))
                warped = warped_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

                concate = np.hstack((raw, target, (warped * 255).astype(np.uint8), flow_vis))
                cv2.putText(concate, 'force:{:4f}'.format(force_value), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 255, 255), 1)
                cv2.putText(concate, 'pose:{:4f}'.format(pose_value), (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 255, 255), 1)
                check_path = 'check/crop'
                image_path = os.path.join(check_path, 'index_{:04d}.png'.format(idx))
                cv2.imwrite(image_path, concate)

        
        return {'raw': raw_tensor,
                'mask': 0,
                'target': target_tensor,
                'flow': flow_tensor,
                'force': force_tensor,
                'pose': pose_tensor,
                'stiffness': stiffness_tensor,
                'raw_path': raw_path
            }
    
          