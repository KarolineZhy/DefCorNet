__author__ = 'yuezhou'

# import basic lib
import cv2
import numpy as np
import os
import sys
import torch
from glob import glob
from PIL import Image
from collections import namedtuple

# import external lib
from core.raft import RAFT
from core.utils.warp import warp

# visualize
from flow_vis import flow_to_color
import matplotlib.pyplot as plt


#######################################################################
## get_flow.py get the groundtruth flow using RAFT
## 
#######################################################################


#######################################################################
############################   Params   ###############################

args = {
'seq_root' :'../framework-master-thesis/data/new_pal_1',
'model_path' : 'raft/raft-sintel.pth',
'small' :False,
'mixed_precision' : True,
'alternate_corr' : False
}
#######################################################################

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory) 


def numpy2tensor(img_np, normalize=True):
    """
    convert np.ndarray to torch.Tensor

    Args:
        img_np (np.ndarray): image, shape [H, W, C] or [H, W]
        normalize (bool, optional): indicate whether normalize the image to [0, 1]. Default True.
    Returns:
        img_tensor (torch.Tensor): image, shape [C, H, W] or [1, H, W]
    """
    assert img_np.ndim == 2 or img_np.ndim == 3
    if img_np.ndim == 2:
        # expand a new dimension
        img_np = img_np[:, :, None]

    # convert to float32
    img_np = img_np.astype(np.float32)

    if normalize:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    return img_tensor

def tensor2numpy(img_tensor):
    """
    Convert tensor to numpy.

    Args:
        img_tensor (torch.Tensor): image. shape: [1, C, H, W] or [C, H, W]
    Return:
        img_np (np.ndarray): image. shape: [H, W, C]
    """
    assert img_tensor.dim() == 3 or img_tensor.dim() == 4
    if img_tensor.dim() == 4:
        assert img_tensor.shape[0] == 1
        img_tensor = img_tensor[0]

    img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    return img_np

if __name__ == '__main__':
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load model
    model = RAFT(args).to(device)
    state_dict = torch.load(args['model_path'])

    # change key since using one gpu, no distribution data parallel
    state_dict = {k.replace('module.', ''): v for (k, v) in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load image
    seq_list = os.listdir(args['seq_root'])

    for seq_idx, seq_name in enumerate(seq_list):
        
        seq_folder = os.path.join(args['seq_root'], seq_name)
        raws = sorted(glob(os.path.join(seq_folder, 'raw/*.png')))
        
        for i,raw in enumerate(raws):
            # load test image
            input = cv2.imread(raw)
            input = cv2.resize(input, (344, 392), cv2.INTER_AREA)
            
            target = cv2.imread(raws[0])
            target = cv2.resize(target, (344, 392), cv2.INTER_AREA)

            target = numpy2tensor(target)[None,:].to(device)
            input = numpy2tensor(input)[None,:].to(device)

            # forward path
            with torch.no_grad():
                _, flow_pr = model(target, input, iters=24, test_mode=True)

            # warp
            warped = warp(input, flow_pr)

            # convert from tensor to numpy
            input_np = tensor2numpy(input)
            target_np = tensor2numpy(target)
            flow_np = tensor2numpy(flow_pr)
            warped_np = tensor2numpy(warped)

            # save each flow
            flow_name = 'flow_{p:04d}_{c:04d}.npy'.format(p=0, c=i)
            
            ## create flow directory
            flow_dir = os.path.join(seq_folder, 'flow')
            createFolder(flow_dir)

            ## define flow name
            flow_name = os.path.join(flow_dir, flow_name)

            ## save flow
            np.save(flow_name, flow_np)
            
            # checking by visualization 
            if True:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                ax1.imshow(input_np, cmap='gray')
                ax1.set_xlabel('input')
                ax2.imshow(target_np, cmap='gray')
                ax2.set_xlabel('target')
                ax3.imshow(warped_np, cmap='gray')
                ax3.set_xlabel('warped')
                ax4.imshow(flow_to_color(flow_np, convert_to_bgr=False))
                ax4.set_xlabel('flow')
                

                flow_vis = flow_to_color(flow_np, convert_to_bgr=False)

                # concate results
                save = np.hstack((input_np*255, target_np*255, warped_np*255, flow_vis*255))
                check_dir = os.path.join('check/flow', str(seq_name))
                createFolder(check_dir)
                
                cv2.imwrite(os.path.join(check_dir, 'flow_{p:04d}_{c:04d}.png'.format(p=i, c=i+1)), save)
            
        print('finish folder: {}'.format(seq_folder))
            
            
print('Finish')
        

