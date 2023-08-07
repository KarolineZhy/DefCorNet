import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


def robust_l1(x):
    """Robust L1 metric"""
    return (x**2 + 0.001**2)**0.5

def image_grads(x, stride=1):
    """Calculate image gradient.
    Args:
        x (torch.Tensor) : dim [B, C, H, W]
    Return:
        dx (torch.Tensor): gradient in direction x, dim [B, C, H, W-1]
        dy (torch.Tensor): gradient in direction y, dim [B, C, H-1, W]
    """
    dx = x[..., stride:] - x[..., :-stride]
    dy = x[..., stride:, :] - x[..., :-stride, :]
    return (dx, dy)

def edge_weighting_fn(x, smoothness_edge_weighting='gaussian', smoothness_edge_constant=150.0):
    """Get weight for image edge.
    Args: 
        x: image input
        smoothness_edge_weighting: 
        smoothness_edge_constant:
    Return:
        weight: dim [B,1,H,W], a weight mask for compution flow smoothness loss
    """
    if smoothness_edge_weighting == 'gaussian':
        return torch.exp(-torch.mean(
            (smoothness_edge_constant*x)**2,
            dim=1,
            keepdim=True))
    elif smoothness_edge_weighting == 'exponential':
        return torch.exp(-torch.mean(
            (abs(smoothness_edge_constant*x)),
            dim=1,
            keepdim=True))
    else:
        raise ValueError('Only gaussian or exponential edge weighting is implemented')


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = F.l1_loss(pred, target)

        return self.loss_weight * loss



@LOSS_REGISTRY.register()
class RobustL1Loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, loss_weight=1.0):
        super(RobustL1Loss, self).__init__()
        self.eps = 1e-6
        self.loss_weight = loss_weight

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class FirstOrderSmoothLoss(nn.Module):
    """First Order Smoothness Loss.
    Args: 
        loss_weight (float): Loss weight for first-order smoothness loss. Default 1.0
        smoothness_edge_weighting (str): smoothness edge weighting. Supported choices are 'gaussian'|'exponential'
        smoothness_dege_constant (float): smoothness edge constant. Default 150.0
    """
    def __init__(self, loss_weight=1.0, smoothness_edge_weighting='gaussian', smoothness_edge_constant=150.0):
        super(FirstOrderSmoothLoss, self).__init__()
        self.loss_weight = loss_weight
        self.smoothness_edge_weighting = smoothness_edge_weighting
        self.smoothness_edge_constant = smoothness_edge_constant

    def forward(self, img, flow):
        """
        Args:
            img (torch.Tensor): img used for edge-aware weighting, dim [B C H W]
            flow (torch.Tensor): Flow field to compu the smoothness loss [B 2 H W]
        """
        img_dx, img_dy = image_grads(img)
        weights_x = edge_weighting_fn(img_dx, self.smoothness_edge_weighting, self.smoothness_edge_constant)
        weights_y = edge_weighting_fn(img_dy, self.smoothness_edge_weighting, self.smoothness_edge_constant)
        flow_dx, flow_dy = image_grads(flow)

        loss_x = torch.mean(weights_x * robust_l1(flow_dx))
        loss_y = torch.mean(weights_y * robust_l1(flow_dy))
        total = loss_x + loss_y

        # compute weighted smoothness loss
        return self.loss_weight * ( total / 2.)

@LOSS_REGISTRY.register()
class SecondOrderSmoothLoss(nn.Module):
    """Second Order Smoothness Loss.
    Args: 
        loss_weight (float): Loss weight for second-order smoothness loss. Default 1.0
        smoothness_edge_weighting (str): smoothness edge weighting. Supported choices are 'gaussian'|'exponential'
        smoothness_dege_constant (float): smoothness edge constant. Default 150.0
    """
    def __init__(self, loss_weight=1.0, smoothness_edge_weighting='gaussian', smoothness_edge_constant=150.0):
        super(SecondOrderSmoothLoss, self).__init__()
        self.loss_weight = loss_weight
        self.smoothness_edge_weighting = smoothness_edge_weighting
        self.smoothness_edge_constant = smoothness_edge_constant

    def forward(self, img, flow):
        """
        Args:
            img (torch.Tensor): img used for edge-aware weighting, dim [B C H W]
            flow (torch.Tensor): Flow field to compu the smoothness loss [B 2 H W]
        """
        img_dx, img_dy = image_grads(img, stride=2)
        weights_xx = edge_weighting_fn(img_dx, self.smoothness_edge_weighting, self.smoothness_edge_constant)
        weights_yy = edge_weighting_fn(img_dy, self.smoothness_edge_weighting, self.smoothness_edge_constant)
        
        # compute second derivatives of the predicted smoothness
        flow_dx, flow_dy = image_grads(flow)
        flow_ddx, _ = image_grads(flow_dx)
        _, flow_ddy = image_grads(flow_dy)

        # compute weighted smoothness
        loss_x = torch.mean(weights_xx * robust_l1(flow_ddx))
        loss_y = torch.mean(weights_yy * robust_l1(flow_ddy))
        total = loss_x + loss_y

        return self.loss_weight * (total / 2.0)