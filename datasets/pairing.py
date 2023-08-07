__author__ = 'yuezhou'

# import basic libs
import numpy as np
import os
from glob import glob
import cv2
import pickle

# import linear assignment related libs
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# import vis module
import matplotlib.pyplot as plt

def linear_assignment(time_stamp_raw, time_stamp_force, time_stamp_pose):
    
    time_stamp_pose = np.array(time_stamp_pose)
    zeros_pose = np.zeros(time_stamp_pose.shape[0])
    time_stamp_pose = np.stack((time_stamp_pose,zeros_pose)).T

    time_stamp_force = np.array(time_stamp_force)
    zeros_force = np.zeros(time_stamp_force.shape[0])
    time_stamp_force = np.stack((time_stamp_force,zeros_force)).T

    time_stamp_raw = np.array(time_stamp_raw)
    zeros_raw = np.zeros(time_stamp_raw.shape[0])
    time_stamp_raw = np.stack((time_stamp_raw,zeros_raw)).T
    
    dist_force_raw = cdist(time_stamp_force, time_stamp_raw, 'euclidean')
    ind_force, ind_raw_1 = linear_sum_assignment(dist_force_raw)
    
    dist_pose_raw = cdist(time_stamp_pose,time_stamp_raw, 'euclidean')
    ind_pose, ind_raw_3 = linear_sum_assignment(dist_pose_raw)
    

    assert ind_raw_1.all() == ind_raw_3.all()
    paired = np.stack((ind_raw_1, ind_force, ind_pose)).T
    
    return paired

def pairing(raw_pkl, mask_pkl, force_pkl, pose_pkl):
    
    # load pickle files
    file_to_read = open(raw_pkl, "rb")
    raw_info_dict = pickle.load(file_to_read)
    file_to_read = open(mask_pkl, "rb")
    mask_info_dict = pickle.load(file_to_read)
    file_to_read = open(force_pkl, "rb")
    force_info_dict = pickle.load(file_to_read)
    file_to_read = open(pose_pkl, "rb")
    pose_info_dict = pickle.load(file_to_read)

    # get timestampe inforation
    raw_ts = raw_info_dict['timestampe']
    mask_ts = mask_info_dict['timestampe']
    force_ts = force_info_dict['timestampe']
    poese_ts = pose_info_dict['timestampe']

    # do linear assigment
    paired = linear_assignment(raw_ts, mask_ts, force_ts, poese_ts)
    
    return paired


#######################################################################
############################   Params   ###############################
data_root = '../data'
#######################################################################

if __name__ == "__main__":
    seq_list = os.listdir(data_root)
    for seq in seq_list:
        seq_dir = os.path.join(data_root, seq)

        # all pkl files has the same name 
        raw_pkl = os.path.join(seq_dir, 'raw_info.pkl')
        mask_pkl = os.path.join(seq_dir, 'mask_info.pkl')
        force_pkl = os.path.join(seq_dir, 'force_info.pkl')
        pose_pkl = os.path.join(seq_dir, 'pose_info.pkl')

        paired = pairing(raw_pkl, mask_pkl, force_pkl, pose_pkl)
        pair_name = 'assignment.npy'

        # save pair information
        pair_path = os.path.join(seq_dir, pair_name)
        np.save(pair_path, paired)