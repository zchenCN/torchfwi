"""
Different kinds of undergroud physical model, i.e rho, vp and vs,
for wave propagation and full waveform inversion

@author: zchen
@date: 2023-02-20
"""
import segyio
import numpy as np
from scipy.interpolate import RegularGridInterpolator
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


def marmousi2(
    nz, nx, 
    path_rho, path_vp, path_vs
):
    # # subsample range, the shape of original data is (13601, 2801) 
    # row_range = range(2500, 12500)
    # col_range = range(500, 2500)
    # rho = tuple()
    # vp = tuple()
    # vs = tuple()
    # with segyio.open(path_rho, strict=False) as f_rho, segyio.open(path_vp, strict=False) as f_vp, segyio.open(path_vs, strict=False) as f_vs:
    #     for row in row_range:
    #         rho += (f_rho.trace[row][col_range], )
    #         vp += (f_vp.trace[row][col_range], )
    #         vs += (f_vs.trace[row][col_range], )
    #     rho = np.vstack(rho).T
    #     vp = np.vstack(vp).T
    #     vs = np.vstack(vs).T

    rho = np.loadtxt(path_rho)
    vp = np.loadtxt(path_vp)
    vs = np.loadtxt(path_vs)
    rho = rho.reshape(150, -1, order='F')
    vp = vp.reshape(150, -1, order='F')
    vs = vs.reshape(150, -1, order='F')


    rho_interp = _subsample(rho, nz, nx)
    vp_interp = _subsample(vp, nz, nx)
    vs_interp = _subsample(vs, nz, nx)
    return rho_interp, vp_interp, vs_interp
    
    

def _subsample(data, nz, nx):
    x, y = np.arange(data.shape[1]), np.arange(data.shape[0])
    interp = RegularGridInterpolator((x, y), data.T)
    nptz = nz + 1
    nptx = nx + 1
    # xx = np.arange(nptx) * ((data.shape[1]-1) // nx)
    # zz = np.arange(nptz) * ((data.shape[0]-1) // nz)
    xx = np.linspace(0, data.shape[1]-1, nptx) 
    zz = np.linspace(0, data.shape[0]-1, nptz)
    X, Z = np.meshgrid(xx, zz)
    xx = X.flatten()[:, np.newaxis]
    zz = Z.flatten()[:, np.newaxis]
    points = np.hstack((xx, zz))
    data_interp = interp(points).reshape(nptz, nptx)
    return data_interp


# def multilayer(nz, nx):
#     def _create_model(nz, nx, v1, v2, v3):
#         nptz, nptx = nz + 1, nx + 1
#         model = np.zeros((nptz, nptx), dtype=np.float32)
#         model[:2*(nptz // 4), :] = v1 
#         model[2*(nptz // 4):3*(nptz // 4), ] = v2
#         model[3*(nptz // 4):] = v3
#         for i in range(nptz // 4, 2*(nptz // 4)):
#             for j in range(nptx):
#                 if j <= nptx//5 or j > 4*nptx//5 or (nptx//5 < j <= 2*nptx//5 and 3*nptz//5-i<=2*nptx//5-j)\
#                     or (3*nptx//5 < j <= 4*nptx//5 and 3*nptz//5-i<=j-3*nptx//5):
#                     model[i, j] = v2
#         return model 
#     rho = _create_model(nz, nx, 1.9, 2.2, 2.6)
#     vp = _create_model(nz, nx, 2.0, 2.6, 3.2)
#     vs = _create_model(nz, nx, 1.2, 2.0, 2.6)
#     return rho, vp, vs

def multilayer():
    def _create_model(v1, v2, v3, v4):
        nptz, nptx = 101, 201
        model = np.zeros((nptz, nptx), dtype=np.float32)
        model[:20] = v1
        model[20:50] = v2
        model[50:85] = v3
        model[85:] = v4
        return model
    rho = _create_model(1.9, 2.1, 2.4, 2.7)
    vp = _create_model(2.0, 2.4, 2.9, 3.2)
    vs = _create_model(1.2, 1.6, 2.1, 2.6)
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
