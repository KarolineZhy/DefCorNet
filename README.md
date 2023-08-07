# DefCor-Net

## Bag Data Preprocessing


### 1. Data Preprocessing
**dataset/data_preprocessing.py** file does data preprocessing tasks, processed the data in defined root, and will return a dict to pair the inforamtion by timestampe, with force, image, pose information.

### 2. Data pairing
**dataset/pairing.py** we can get the paired image with force and pose by linear assignment from timestamp.

### 3. Genearate ground truth

**raft/generate_gt_stepwise.py** will generate gt flow between 2 frames. When the deformation is too large, we will use an intermidiate frame to do a step-wise flow estimation.

## Network Details
<p align="center">
<img src="src/overview.png" alt="drawing" width="70%" />
</p>

The code implementation of the network see **networks/c2f_stiff_second_network.py**

## How to use
```
python train.py --opt options/train_c2f_stiff_second.yaml
```