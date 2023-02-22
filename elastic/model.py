"""
Different kinds of undergroud physical model, i.e rho, vp and vs,
for wave propagation and full waveform inversion

@author: zchen
@date: 2023-02-20
"""

import numpy as np
import torch

def homogenenous(
    nz, nx, 
    rho=2.0, vp=4.7, vs=3.5
):
    rho = rho * np.ones((nz+1, nx+1), dtype=np.float32)
    vp = vp * np.ones((nz+1, nx+1), dtype=np.float32)
    vs = vs * np.ones((nz+1, nx+1), dtype=np.float32)
    return rho, vp, vs 


def square(
    nz, nx,
    rho=1.5, vp=4.0, vs=3.0
):
    rho = rho * np.ones((nz+1, nx+1), dtype=np.float32)
    rho[nz//4:3*nz//4, nx//4:3*nz//4] += 0.3
    vp = vp * np.ones((nz+1, nx+1), dtype=np.float32)
    vp[nz//4:3*nz//4, nx//4:3*nz//4] += 0.5
    vs = vs * np.ones((nz+1, nx+1), dtype=np.float32)
    vs[nz//4:3*nz//4, nx//4:3*nz//4] += 0.4
    return rho, vp, vs 

            

def box_smooth(model, h=20, w=20):
    """
    Smooth the standard model to create initial model

    Parameters:
    -----------
    h: int
        Hight of box for average pooling
    w: int
        Width of box for average pooling
    """
    # pading with edge values
    padding_left = (w - 1) // 2
    padding_right = w - 1 - padding_left
    padding_top = (h - 1) // 2
    padding_bottom = h - 1 - padding_top
    m = torch.nn.ReplicationPad2d((padding_left, padding_right, padding_top, padding_bottom))
    init = model.unsqueeze(0).unsqueeze(0)
    init = m(init)
    # kernel
    kernel = torch.ones(h, w) / (h*w)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    init = torch.nn.functional.conv2d(init, kernel).squeeze()
    assert init.shape == model.shape 
    return init
