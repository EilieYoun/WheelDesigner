import numpy as np 
import torch 
from torch.nn import functional as F
from WheelDesigner.data_utils import rotate

def get_rec_loss(x, x_hat, divide=None):
    loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    if divide is not None: loss/=divide
    return loss

def get_kl_loss(mu, log_var, divide=None):
    loss = 1 + log_var - mu.pow(2) - log_var.exp()
    loss = torch.mean(torch.sum(loss))
    loss *= -0.5
    if divide is not None: loss/=divide
    return loss


def get_rot_loss(x_hat, y, divide=None):
    x_hat_cp = x_hat.detach().cpu().numpy()
    x_hat_cp = np.where(x_hat_cp > 0.5, 1.0, 0.0) 
    y_cp = y.detach().cpu().numpy()

    x_hat_rots = list(map(lambda xi, yi: rotate(xi, yi), x_hat_cp, y_cp))
    diffs = []
    for i in range(len(x_hat_cp)):
        diff = np.sum(np.abs(x_hat_cp[i] - x_hat_rots[i]))
        diffs.append(diff)
        
    loss = sum(diffs)
    if divide is not None: loss/=divide
    return loss
