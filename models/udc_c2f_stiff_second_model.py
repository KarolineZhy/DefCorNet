from logging import root
from cv2 import norm
import torch
import torch.nn.functional as F

from losses import build_loss

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.flow_util import warp
from utils.data_util import tensor2numpy

from flow_vis import flow_to_color
import matplotlib.pyplot as plt
from os.path import dirname as up
import numpy as np
import cv2
import os
import time

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


@MODEL_REGISTRY.register()
class CorseToFineStiffSecondModel(BaseModel):

    def __init__(self, opt):
        super(CorseToFineStiffSecondModel, self).__init__(opt=opt)
        self.opt = opt

    def feed_data(self, data):
        """process data"""
        # move data to device
        for k, v in data.items():
            if not isinstance(v, list):
                data[k] = v.to(self.device)

        # load data
        raw = data['raw']
        force = data['force']
        target_flow = data['flow']
        target = data['target']
        stiffness = data['stiffness']

        # concat input with force
        input_cat = torch.cat((raw, force, stiffness), dim=1)
        
        # forward path
        flow_list, mask_list = self.networks['flow_net'](input_cat)
        # calculate loss
        self.loss_metrics = self._calculate_loss(raw, flow_list, target_flow)

        # get visualization
        self._get_visualization(raw, target, target_flow, force, flow_list[2], mask_list)
        


    def optimize_parameters(self):
        """forward pass"""
        loss = self.loss_metrics['total_loss']

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        loss.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()
    
    def _setup_losses(self):
        train_opt = self.opt['train']
        self.l1_loss = build_loss(train_opt['losses']['l1'])
        self.first_order_smooth_loss = build_loss(train_opt['losses']['first_smooth'])
        self.second_order_smooth_loss = build_loss(train_opt['losses']['second_smooth'])

    def _calculate_loss(self, img, flow_list, gt_flow):
        """Compute different loss for back propagation.
        Args:
            img (torch.Tensor): deformed image, dim [B, C, H, W]
            pred_flow (torch.Tensor): predicted flow, dim [B, 2, H, W]
            gt_flow (torch.Tensor): ground truth flow, dim [B, 2, H, W]
        Returns:
            loss_metrics (dict): dictionary holding the calculated losses.
        """
        
        # get different scale gt_flow
        gt_flow_3 = gt_flow
        gt_flow_2 = F.interpolate(gt_flow_3, scale_factor=0.5) * 0.5
        gt_flow_1 = F.interpolate(gt_flow_2, scale_factor=0.5) * 0.5

        # Initialize losses to be zero
        metrics = {}
        metrics['total_loss'] = 0.0
        if self.l1_loss.loss_weight > 0:
            metrics['l1_flow1_loss'] = 0.0
            metrics['l1_flow2_loss'] = 0.0
            metrics['l1_flow3_loss'] = 0.0
        if self.first_order_smooth_loss.loss_weight > 0:
            metrics['first_order_smooth_loss'] = 0.0
        if self.second_order_smooth_loss.loss_weight > 0:
            metrics['second_order_smooth_loss'] = 0.0 

        # L1 loss
        if self.l1_loss.loss_weight > 0:
            l1_loss = self.l1_loss(flow_list[0], gt_flow_1)
            metrics['total_loss'] += l1_loss
            metrics['l1_flow1_loss'] = l1_loss

            l1_loss = self.l1_loss(flow_list[1], gt_flow_2)
            metrics['total_loss'] += l1_loss
            metrics['l1_flow2_loss'] = l1_loss

            l1_loss = self.l1_loss(flow_list[2], gt_flow_3)
            metrics['total_loss'] += l1_loss
            metrics['l1_flow3_loss'] = l1_loss
            
        # first order smoothness loss
        if self.first_order_smooth_loss.loss_weight > 0:
            first_order_smooth_loss = self.first_order_smooth_loss(img, flow_list[2])
            metrics['total_loss'] += first_order_smooth_loss
            metrics['first_order_smooth_loss'] = first_order_smooth_loss

        # second order smoothness loss
        if self.second_order_smooth_loss.loss_weight > 0:
            second_order_smooth_loss = self.second_order_smooth_loss(img, flow_list[2])
            metrics['total_loss'] += second_order_smooth_loss
            metrics['second_order_smooth_loss'] = second_order_smooth_loss

        return metrics
    
    # def _get_visualization(self, input, target, target_flow, force, flow, mask_lo=None, mask_up=None):
    #     warped = warp(input, flow)
    #     target_warped = warp(input, target_flow)
    #     self.vis = {
    #         'raw' : tensor2numpy(input),
    #         'target': tensor2numpy(target),
    #         'warped': tensor2numpy(warped),
    #         'target_warped': tensor2numpy(target_warped),
    #         'flow': tensor2numpy(flow),
    #         'force': np.max(tensor2numpy(force)) * 10 ,
    #         'local': tensor2numpy(mask_lo),
    #         'update': tensor2numpy(mask_up)
    #         }
    def _get_visualization(self, input, target, target_flow, force, flow, mask_list):
        warped = warp(input, flow)
        target_warped = warp(input, target_flow)
        mask_3 = mask_list[0]
        mask_2 = mask_list[1]
        mask_1 = mask_list[2]
        mask_1_unet = mask_list[3]
        mask_2_unet = mask_list[4]
        mask_3_unet = mask_list[5] 
        stiff_1_update = mask_list[6]
        stiff_2_update = mask_list[7]
        stiff_3_update = mask_list[8]


        self.vis = {
            'raw' : tensor2numpy(input),
            'target': tensor2numpy(target),
            'warped': tensor2numpy(warped),
            'target_warped': tensor2numpy(target_warped),
            'flow': tensor2numpy(flow),
            'force': np.max(tensor2numpy(force)) * 10,
            'mask_1_vis': tensor2numpy((mask_1-torch.min(mask_1))/(torch.max(mask_1)-torch.min(mask_1))),
            'mask_2_vis': tensor2numpy((mask_2-torch.min(mask_2))/(torch.max(mask_2)-torch.min(mask_2))),
            'mask_3_vis': tensor2numpy((mask_3-torch.min(mask_3))/(torch.max(mask_3)-torch.min(mask_3))),
            'mask_1': tensor2numpy(mask_1),
            'mask_2': tensor2numpy(mask_2),
            'mask_3': tensor2numpy(mask_3),
            'mask_1_unet': tensor2numpy(mask_1_unet),
            'mask_2_unet': tensor2numpy(mask_2_unet),
            'mask_3_unet': tensor2numpy(mask_3_unet),
            'stiff_1_update': tensor2numpy(stiff_1_update),
            'stiff_2_update': tensor2numpy(stiff_2_update),
            'stiff_3_update': tensor2numpy(stiff_3_update)
            }
        
    def _display_plots(self, tb_logger, mode):

        # display
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1, 8, figsize=(20, 5))
        plt.title('Force {} N'.format(self.vis['force']))
        ax1.imshow(self.vis['raw'], cmap='gray')
        ax1.set_xlabel('input')
        ax2.imshow(self.vis['target'], cmap='gray')
        ax2.set_xlabel('target')
        ax3.imshow(self.vis['warped'], cmap='gray')
        ax3.set_xlabel('warped')
        ax4.imshow(self.vis['target_warped'], cmap='gray')
        ax4.set_xlabel('gt_warp')
        ax5.imshow(flow_to_color(self.vis['flow'], convert_to_bgr=False))
        ax5.set_xlabel('flow')
        ax6.imshow(self.vis['mask_1'], cmap='gray')
        ax6.set_xlabel('mask_1_vis')
        ax7.imshow(self.vis['mask_2'], cmap='gray')
        ax7.set_xlabel('mask_2_vis')
        ax8.imshow(self.vis['mask_3'], cmap='gray')
        ax8.set_xlabel('mask_3_vis')


        if tb_logger:
            tb_logger.add_figure(mode, fig, self.curr_iter)

    def train_plot(self, tb_logger):
        self._display_plots(tb_logger=tb_logger, mode='Training')


    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # validation
        self.eval()

        # get data
        data = next(iter(dataloader))

        # move to device
        for k, v in data.items():
            if not isinstance(v, list):
                data[k] = v.to(self.device)

        raw = data['raw']
        force = data['force']
        target_flow = data['flow']
        target = data['target']
        stiffness = data['stiffness']

        # forward pass
        with torch.no_grad():
            flow_list, mask_list = self.networks['flow_net'](torch.cat((raw, force, stiffness), dim=1))

        self._get_visualization(raw, target, target_flow, force, flow_list[2], mask_list)

        self._display_plots(tb_logger, 'Validation')

    def test(self, test_loader):

        self.eval()

        # get data
        for i, data in enumerate(test_loader):
           
            # move to device
            for k, v in data.items():
                if not isinstance(v, list):
                    data[k] = v.to(self.device)

            raw = data['raw']
            force = data['force']
            target_flow = data['flow']
            target = data['target']
            raw_path = data['raw_path']
            stiffness = data['stiffness']
            
            
            # forward pass
            
            with torch.no_grad():
                flow_list, mask_list = self.networks['flow_net'](torch.cat((raw, force, stiffness), dim=1))
            
            
            self._get_visualization(raw, target, target_flow, force, flow_list[2],mask_list)

            concate = np.hstack((self.vis['raw'], self.vis['target'], self.vis['warped'], self.vis['target_warped']))
            concate = np.uint8(concate*255)


            if self.opt['path']['visualization'] is None:
                root_path = self.opt['path']['experiments_root']
            else:
                root_path = self.opt['path']['visualization']
         
            seq_name = up(up(raw_path[0])).split('/')[-1]
            seq_path = os.path.join(root_path, seq_name)
            sample_idx = os.path.basename(raw_path[0]).split('.')[0]
            sample_idx = sample_idx.split('_')[1]
            
            # visualization
            createFolder(seq_path)
            img_name_saved = os.path.join(seq_path, 'result_{}.png'.format(sample_idx))
            cv2.imwrite(img_name_saved, concate)
            

            mask_local_name_saved = os.path.join(seq_path, 'mask_1_{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['mask_1'])
            mask_local_name_saved = os.path.join(seq_path, 'mask_2_{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['mask_2'])
            mask_local_name_saved = os.path.join(seq_path, 'mask_3_{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['mask_3'])
            mask_local_name_saved = os.path.join(seq_path, 'mask_1_unet_{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['mask_1_unet'])
            mask_local_name_saved = os.path.join(seq_path, 'mask_2_unet_{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['mask_2_unet'])
            mask_local_name_saved = os.path.join(seq_path, 'mask_3_unet_{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['mask_3_unet'])
            mask_local_name_saved = os.path.join(seq_path, 'stiff_1_update{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['stiff_1_update'])
            mask_local_name_saved = os.path.join(seq_path, 'stiff_2_update{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['stiff_2_update'])
            mask_local_name_saved = os.path.join(seq_path, 'stiff_3_update{}.npy'.format(sample_idx))
            np.save(mask_local_name_saved, self.vis['stiff_3_update'])
            
           
            # predictied flow
            root_path = self.opt['path']['flow']
            seq_path = os.path.join(root_path, seq_name)
            createFolder(seq_path)
            
            flow_name_saved = os.path.join(seq_path, 'pred_{}.npy'.format(sample_idx))
            np.save(flow_name_saved,self.vis['flow'])
    
    def test_anatomy(self, test_loader):

        self.eval()

        # get data
        for epoch in np.arange(10):
            for i, data in enumerate(test_loader):
            
                # move to device
                for k, v in data.items():
                    if not isinstance(v, list):
                        data[k] = v.to(self.device)

                raw = data['raw']
                force = data['force']
                target_flow = data['flow']
                target = data['target']
                raw_path = data['raw_path']
                stiffness = data['stiffness']
                
                
                # forward pass
                
                with torch.no_grad():
                    flow_list, mask_list = self.networks['flow_net'](torch.cat((raw, force, stiffness), dim=1))
                
                
                self._get_visualization(raw, target, target_flow, force, flow_list[2],mask_list)
              

                concate = np.hstack((np.uint8(self.vis['raw']*255), 
                                     np.uint8(self.vis['target']*255), 
                                     np.uint8(self.vis['warped']*255), 
                                     np.uint8(self.vis['target_warped']*255), 
                                     flow_to_color(self.vis['flow'], convert_to_bgr=True)))
  
            


                if self.opt['path']['visualization'] is None:
                    root_path = self.opt['path']['experiments_root']
                else:
                    root_path = self.opt['path']['visualization']
            
                seq_name = up(up(raw_path[0])).split('/')[-1]
                seq_path = os.path.join(root_path, seq_name)
                sample_idx = os.path.basename(raw_path[0]).split('.')[0]
                sample_idx = sample_idx.split('_')[1]
                
                # visualization
                createFolder(seq_path)
                img_name_saved = os.path.join(seq_path, 'result_{}_{}.png'.format(sample_idx, epoch))
                cv2.imwrite(img_name_saved, concate)

                
            
                # predictied flow
                root_path = self.opt['path']['flow']
                seq_path = os.path.join(root_path, seq_name)
                createFolder(seq_path)
                
                flow_name_saved = os.path.join(seq_path, 'pred_{}_{}.npy'.format(sample_idx, epoch))
                np.save(flow_name_saved,self.vis['flow'])

                
           

