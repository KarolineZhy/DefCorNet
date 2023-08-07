__author__ = 'yuezhou'

# import basic lib 
from unittest import result
import rosbag
import pickle
import os
from glob import glob
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


# import external lib
from pairing import linear_assignment
import argparse

#######################################################################
## data_preprocess.py used to extract bag file information to a dict 
## saved folder root
## -data
##   - sequence
##     - force info dict force_info.pkl
##     - raw info dict raw_info.pkl
##     - pose info dict pose_info.pkl
##     - image folder
##       - raw
##         - raw_0000.png
##         - raw_0001.png
##         - ...
##       - flow
##         - flow_0000_0000.npy
##         - flow_0000_0001.npy
##         - flow_0000_0002.npy
#######################################################################


#######################################################################
############################   Params   ###############################
bag_root = '/home/eadu/workspace/yue/data/same_point/'
ros_topic_force = '/ForceSensor/force'
ros_topic_raw = '/jzl/raw'
ros_topic_mask = '/jzl/mask'
#######################################################################

def initialize_dict():
    dict_force_info = {
        'timestampe':[],
        'force':{
            'x':[],
            'y':[],
            'z':[]
        }
    }

    dict_pose_info = {
        'timestampe':[],
        'position':{
            'x':[],
            'y':[],
            'z':[]
        },
        'orientation':{
            'x':[],
            'y':[],
            'z':[],
            'w':[]
        }

    }

    dict_raw_info ={
        'timestampe':[],
        'raw_path':[],
    }

    dict_mask_info ={
        'timestampe':[],
        'mask_path':[],
    }
    return dict_raw_info, dict_mask_info, dict_force_info, dict_pose_info

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def main(args): 
    # list all bag files
    bag_files = sorted(glob(os.path.join(args.dataroot, '*.bag')))
    topics_image = ["/imfusion/cephasonics"]

    # loop over each bag file
    for file in bag_files:
        sequence_filename = os.path.splitext(os.path.basename(file))[0]
       
        saved_directory = os.path.join(args.saveroot, sequence_filename)
        
        # create directory
        createFolder(saved_directory)
        dict_raw_info, dict_mask_info, dict_force_info, dict_pose_info = initialize_dict()
        
        bag = rosbag.Bag(file, "r")

        count = 0
        for topic, msg, t in bag.read_messages(topics="/imfusion/cephasonics"):
            
            saved_directory_raw = os.path.join(saved_directory, 'raw')
            createFolder(saved_directory_raw)

            # read raw info         
            format_data = np.fromstring(msg.data, dtype=np.uint8).reshape(1080,1920)
            format_data = format_data[120:806, 580:1351]
            format_data = cv2.resize(format_data, (320, 384), cv2.INTER_AREA)
            timestampe = msg.header.stamp 
           

            output_name = 'raw_{0:04d}.png'.format(count)
            output_path = os.path.join(saved_directory_raw, output_name)
            
            # add to raw info dict
            dict_raw_info['timestampe'].append(timestampe.to_nsec())
            dict_raw_info['raw_path'].append(output_name)


            # save raw image
            cv2.imwrite(output_path, format_data) 

            # save raw dict
            raw_file = open(os.path.join(saved_directory, 'raw_info.pkl'), "wb")
            pickle.dump(dict_raw_info, raw_file)
            raw_file.close()

            count += 1

            
        for topic, msg, t in bag.read_messages(topics=["/ForceSensor/force"]):
        
            # add force info to dice
            dict_force_info['timestampe'].append(t.to_nsec())
            dict_force_info['force']['x'].append(msg.wrench.force.x)
            dict_force_info['force']['y'].append(msg.wrench.force.y)
            dict_force_info['force']['z'].append(msg.wrench.force.z)

            # save force info dict
            force_file = open(os.path.join(saved_directory, 'force_info.pkl'), "wb")
            pickle.dump(dict_force_info, force_file)
            force_file.close()   

        for topic, msg, t in bag.read_messages(topics=["/iiwa/state/CartesianPose"]):
        
            # add pose info to dice
            dict_pose_info['timestampe'].append(t.to_nsec())
            dict_pose_info['position']['x'].append(msg.poseStamped.pose.position.x)
            dict_pose_info['position']['y'].append(msg.poseStamped.pose.position.y)
            dict_pose_info['position']['z'].append(msg.poseStamped.pose.position.z)
            
            dict_pose_info['orientation']['x'].append(msg.poseStamped.pose.orientation.x)
            dict_pose_info['orientation']['y'].append(msg.poseStamped.pose.orientation.y)
            dict_pose_info['orientation']['z'].append(msg.poseStamped.pose.orientation.z)
            dict_pose_info['orientation']['w'].append(msg.poseStamped.pose.orientation.w)

            # save pose info dict
            pose_file = open(os.path.join(saved_directory, 'pose_info.pkl'), "wb")
            pickle.dump(dict_pose_info, pose_file)
            pose_file.close() 

        # get all timestampe  for each sensor information
        raw_ts = dict_raw_info['timestampe']
        force_ts = dict_force_info['timestampe'] 
        pose_ts = dict_pose_info['timestampe']

        # using linear assignment to pair raw, force, pose timestampe
        paired_res = linear_assignment(raw_ts, force_ts, pose_ts)
        
        # initialize an empty dict for saving data pairs
        result_dict = {}

        # using raw image path as the key, and related info save in the value
        for r,f,p in paired_res:
            
            result_dict[dict_raw_info['raw_path'][r]] = {
               
                'force': {
                    'x': dict_force_info['force']['x'][f],
                    'y': dict_force_info['force']['y'][f],
                    'z': dict_force_info['force']['z'][f]
                },

                'pose': {
                    'position': {
                        'x' : dict_pose_info['position']['x'][p],
                        'y' : dict_pose_info['position']['y'][p],
                        'z' : dict_pose_info['position']['z'][p]
                    },
                    'orientation':{
                        'x' : dict_pose_info['orientation']['x'][p],
                        'y' : dict_pose_info['orientation']['y'][p],
                        'z' : dict_pose_info['orientation']['z'][p],
                        'w' : dict_pose_info['orientation']['w'][p],
                    }

                }
            }

        # get stiffness
        st_force = []
        st_pose = []
        for k,v in result_dict.items():
            st_force.append(v['force']['z']*1)
            st_pose.append(v['pose']['position']['z']*1000)

        coef = np.polyfit(st_pose, st_force, 1)

        for k,v in result_dict.items():
            result_dict[k]['stiffness'] = coef[0]
    
        
        # save final result dict
        result_file = open(os.path.join(saved_directory, 'result_dict.pkl'), "wb")
        pickle.dump(result_dict, result_file)
        result_file.close() 
        print('Finish: ' + file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--dataroot', type=str, help='root for to be processed ros file ')
    parser.add_argument('--saveroot', type=str, help='ros file saved root')

    args = parser.parse_args()
    main(args)
