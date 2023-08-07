import torch
import torch.nn.functional as F


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def flow_to_warp(flow):
    """
    Compute the warp from the flow field
    Args:
        flow: optical flow shape [B, 2, H, W]
    Returns:
        warp: the endpoints of the estimated flow. shape [B, H, W, 2]
    """
    flow = flow.permute(0, 2, 3, 1)
    B, H, W, _ = flow.size()
    grid = coords_grid(B, H, W)
    if flow.is_cuda:
        grid = grid.cuda()
    grid = grid.permute(0, 2, 3, 1)
    warp = grid + flow
    return warp

def warp(x, flo, mode='bilinear'):
    H, W = flo.shape[-2:]
    vgrid = flow_to_warp(flo)
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W-1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H-1, 1) - 1.0
    if mode == 'bilinear':
        output = F.grid_sample(x, vgrid, mode=mode, align_corners=True)
    else:
        output = F.grid_sample(x, vgrid, mode=mode)
    return output