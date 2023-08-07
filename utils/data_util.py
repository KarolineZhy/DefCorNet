import numpy as np
import torch
import yaml


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
        # assert img_tensor.shape[0] == 1
        img_tensor = img_tensor[0]

    img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    return img_np


def numpy2tensor(img_np, normalize=True, max_force=0):
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
        if max_force !=0:
            img_np = img_np / max_force
        else:
            img_np = img_np / 255.

    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    return img_tensor


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
