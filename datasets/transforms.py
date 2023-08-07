import random
import numpy as np

def random_flip_left_right(img, target, flow, probability_flip_left_right):
    """Performs a random left/right flip."""
    perform_flip = random.random() < probability_flip_left_right

    if perform_flip:
        img = img[:, ::-1]
        target = target[:, ::-1]

        if flow is not None:
            flow = flow[:, ::-1]
            # correct sign of flow
            flow[:, :, 0] *= -1

    return img, target, flow

def random_crop_vertical(img, target, flow, force, stiffness, probability_crop_vertical, crop_width=200, is_fill=True):
    """Performs a vertical crop, which preserves the height"""
    perfrom_crop = random.random() < probability_crop_vertical

    if perfrom_crop:
        # select width range
        W = img.shape[1]
        start_pix = int(random.random() * (W - crop_width))
        crop_range = [start_pix, start_pix+crop_width]

        # cropping
        if is_fill:
            # fill zero to cropped area size [H * W]
            zero_img = np.zeros_like(img)
            zero_tar = np.zeros_like(target)
            zero_flow = np.zeros_like(flow)
            zero_for = np.zeros_like(force)
            zero_stiff = np.zeros_like(stiffness)

            # add image to zeros
            zero_img[:, crop_range[0]: crop_range[1], :] = img[:, crop_range[0]: crop_range[1], :]
            zero_tar[:, crop_range[0]: crop_range[1], :] = target[:, crop_range[0]: crop_range[1], :]
            zero_for[:, crop_range[0]: crop_range[1], :] = force[:, crop_range[0]: crop_range[1], :]
            zero_flow[:, crop_range[0]: crop_range[1], :] = flow[:, crop_range[0]: crop_range[1], :]
            zero_stiff[:, crop_range[0]: crop_range[1], :] = force[:, crop_range[0]: crop_range[1], :]

            # return images
            img = zero_img
            target = zero_tar
            flow = zero_flow
            force = zero_for
            stiffness = zero_stiff
            
        else:
            # resize to [H * crop_width]
            img = img[:, crop_range[0]: crop_range[1], :]
            target = target[:, crop_range[0]: crop_range[1], :]
            force = force[:, crop_range[0]: crop_range[1], :]
            stiffness = stiffness[:, crop_range[0]: crop_range[1], :]
            flow = flow[:, crop_range[0]: crop_range[1], :]

    return img, target, flow, force, stiffness

def apply_transform(raw, target, flow, force, stiffness, args):
    flip_left_right_opt = args.get('flip_left_right', None)
    crop_vertical_opt = args.get('crop_vertical', None)
    raw, target, flow = random_flip_left_right(raw, target, flow, **flip_left_right_opt)
    raw, target, flow, force, stiffness = random_crop_vertical(raw, target, flow, force, stiffness, **crop_vertical_opt)
        
    return raw, target, flow, force, stiffness