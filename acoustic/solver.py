"""
Finite difference solver for 2d acoustic wave equations
with perfectly matched layer. Here the code is implemented by PyTorch,
for auto differentiation.

For more details about the PML and finite difference method,
please refernce this paper: https://hal.inria.fr/inria-00073219/file/RR-3471.pdf

@author: zchen
@date: 2023-03-06
"""

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

def ricker(dt, nt, peak_time, dominant_freq):
    """Ricker wavelet with specific dominant frequency"""
    t = np.arange(-peak_time, dt * nt - peak_time, dt, dtype=np.float32)
    w = ((1.0 - 2.0 * (np.pi**2) * (dominant_freq**2) * (t**2))
        * np.exp(-(np.pi**2) * (dominant_freq**2) * (t**2)))
    return w * 1e6


class Solver:
    def __init__(self, model,
                h, dt, nt, peak_time, dominant_freq,
                receivers_xz, 
                pml_width=10, pad_width=10):
        
        assert model.dtype == torch.float32

        # Device 
        self.device = model.device      

        # Mesh
        self.nptz, self.nptx = model.shape[0], model.shape[1]
        self.nt = int(nt)
        self.h = float(h)
        self.dt = float(dt)

        # Sources and receivers
        self.receivers_xz = receivers_xz

        # Source time function
        self.peak_time = peak_time 
        self.dominant_freq = dominant_freq
        source_time = ricker(self.dt, self.nt, self.peak_time, self.dominant_freq)
        self.source_time = torch.tensor(source_time, device=self.device, dtype=torch.float32)

        # PML and padding
        self.pml_width = pml_width 
        self.pad_width = pad_width 
        self.total_pad = pml_width + pad_width
        self.nptx_padded = self.nptx + 2 * self.total_pad 
        self.nptz_padded = self.nptz + 2 * self.total_pad

        # Model 
        self.model = model

        # Dampling factors 
        self._calc_dampling_factors()

    def _calc_dampling_factors(self):
        # dampling factor
        profile = 40.0 + 60.0 * np.arange(self.pml_width, dtype=np.float32)

        sigma_x = np.zeros(self.nptx_padded, np.float32)
        sigma_x[self.total_pad-1:self.pad_width-1:-1] = profile 
        sigma_x[-self.total_pad:-self.pad_width] = profile
        sigma_x[:self.pad_width] = sigma_x[self.pad_width]
        sigma_x[-self.pad_width:] = sigma_x[-self.pad_width-1]
        sigma_x = np.tile(sigma_x, (self.nptz_padded, 1))

        sigma_z = np.zeros(self.nptz_padded, np.float32)
        sigma_z[self.total_pad-1:self.pad_width-1:-1] = profile 
        sigma_z[-self.total_pad:-self.pad_width] = profile
        sigma_z[:self.pad_width] = sigma_z[self.pad_width]
        sigma_z[-self.pad_width:] = sigma_z[-self.pad_width-1]
        sigma_z = np.tile(sigma_z.reshape(-1, 1), (1, self.nptx_padded))

        self.sigma_x = torch.tensor(sigma_x, dtype=torch.float32, device=self.device)
        self.sigma_z = torch.tensor(sigma_z, dtype=torch.float32, device=self.device)


    def _set_model(self):
        # Padding
        pad = nn.ReplicationPad2d((self.total_pad,) * 4)
        model_padded = pad(self.model[None, :]).squeeze()
        return model_padded 
    
    def first_x_deriv(self, f):
        fx = (5 * f[:, :, self.pad_width-6:-self.pad_width-6]
                - 72 * f[:, :, self.pad_width-5:-self.pad_width-5]
                + 495 * f[:, :, self.pad_width-4:-self.pad_width-4]
                - 2200 * f[:, :, self.pad_width-3:-self.pad_width-3]
                + 7425 * f[:, :, self.pad_width-2:-self.pad_width-2]
                - 23760 * f[:, :, self.pad_width-1:-self.pad_width-1]
                + 23760 * f[:, :, self.pad_width+1:-self.pad_width+1]
                - 7425 * f[:, :, self.pad_width+2:-self.pad_width+2]
                + 2200 * f[:, :, self.pad_width+3:-self.pad_width+3]
                - 495 * f[:, :, self.pad_width+4:-self.pad_width+4]
                + 72 * f[:, :, self.pad_width+5:-self.pad_width+5]
                - 5 * f[:, :, self.pad_width+6:-self.pad_width+6]) / (27720*self.h)
        fx = F.pad(fx, (self.pad_width, self.pad_width), "constant", 0)
        assert fx.shape[1] == self.nptz_padded and fx.shape[2] == self.nptx_padded 
        return fx 
    
    def first_z_deriv(self, f):
        fz = (5 * f[:, self.pad_width-6:-self.pad_width-6, :]
                - 72 * f[:, self.pad_width-5:-self.pad_width-5, :]
                + 495 * f[:, self.pad_width-4:-self.pad_width-4, :]
                - 2200 * f[:, self.pad_width-3:-self.pad_width-3, :]
                + 7425 * f[:, self.pad_width-2:-self.pad_width-2, :]
                - 23760 * f[:, self.pad_width-1:-self.pad_width-1, :]
                + 23760 * f[:, self.pad_width+1:-self.pad_width+1, :]
                - 7425 * f[:, self.pad_width+2:-self.pad_width+2, :]
                + 2200 * f[:, self.pad_width+3:-self.pad_width+3, :]
                - 495 * f[:, self.pad_width+4:-self.pad_width+4, :]
                + 72 * f[:, self.pad_width+5:-self.pad_width+5, :]
                - 5 * f[:, self.pad_width+6:-self.pad_width+6, :]) / (27720*self.h)
        fz = F.pad(fz, (0, 0, self.pad_width, self.pad_width), "constant", 0)
        assert fz.shape[1] == self.nptz_padded and fz.shape[2] == self.nptx_padded 
        return fz 
    
    def second_x_deriv(self, f):
        """Second derivative of f with respect to x"""
        fxx = (- 735 * f[:, :, self.pad_width-8:-self.pad_width-8]
                + 15360 * f[:, :, self.pad_width-7:-self.pad_width-7]
                -  156800 * f[:, :, self.pad_width-6:-self.pad_width-6]
                + 1053696 * f[:, :, self.pad_width-5:-self.pad_width-5]
                - 5350800 * f[:, :, self.pad_width-4:-self.pad_width-4]
                + 22830080 * f[:, :, self.pad_width-3:-self.pad_width-3]
                - 94174080 * f[:, :, self.pad_width-2:-self.pad_width-2]
                + 538137600 * f[:, :, self.pad_width-1:-self.pad_width-1]
                - 924708642 * f[:, :, self.pad_width:-self.pad_width]
                + 538137600 * f[:, :, self.pad_width+1:-self.pad_width+1]
                - 94174080 * f[:, :, self.pad_width+2:-self.pad_width+2]
                + 22830080 * f[:, :, self.pad_width+3:-self.pad_width+3]
                - 5350800 * f[:, :, self.pad_width+4:-self.pad_width+4]
                + 1053696 * f[:, :, self.pad_width+5:-self.pad_width+5]
                - 156800 * f[:, :, self.pad_width+6:-self.pad_width+6]
                + 15360 * f[:, :, self.pad_width+7:-self.pad_width+7]
                - 735 * f[:, :, self.pad_width+8:-self.pad_width+8]
                ) / (302702400*self.h**2)
        fxx = F.pad(fxx, (self.pad_width, self.pad_width), "constant", 0)
        assert fxx.shape[1] == self.nptz_padded and fxx.shape[2] == self.nptx_padded 
        return fxx  
    
    def second_z_deriv(self, f):
        """Second derivative of f with respect to z"""
        fzz = (- 735 * f[:, self.pad_width-8:-self.pad_width-8, :]
                + 15360 * f[:, self.pad_width-7:-self.pad_width-7, :]
                -  156800 * f[:, self.pad_width-6:-self.pad_width-6, :]
                + 1053696 * f[:, self.pad_width-5:-self.pad_width-5, :]
                - 5350800 * f[:, self.pad_width-4:-self.pad_width-4, :]
                + 22830080 * f[:, self.pad_width-3:-self.pad_width-3, :]
                - 94174080 * f[:, self.pad_width-2:-self.pad_width-2, :]
                + 538137600 * f[:, self.pad_width-1:-self.pad_width-1, :]
                - 924708642 * f[:, self.pad_width:-self.pad_width, :]
                + 538137600 * f[:, self.pad_width+1:-self.pad_width+1, :]
                - 94174080 * f[:, self.pad_width+2:-self.pad_width+2, :]
                + 22830080 * f[:, self.pad_width+3:-self.pad_width+3, :]
                - 5350800 * f[:, self.pad_width+4:-self.pad_width+4, :]
                + 1053696 * f[:, self.pad_width+5:-self.pad_width+5, :]
                - 156800 * f[:, self.pad_width+6:-self.pad_width+6, :]
                + 15360 * f[:, self.pad_width+7:-self.pad_width+7, :]
                - 735 * f[:, self.pad_width+8:-self.pad_width+8, :]
                ) / (302702400*self.h**2)
        fzz = F.pad(fzz, (0, 0, self.pad_width, self.pad_width), "constant", 0)
        assert fzz.shape[1] == self.nptz_padded and fzz.shape[2] == self.nptx_padded 
        return fzz  
    
    def laplacian(self, f):
        return self.second_x_deriv(f) + self.second_z_deriv(f)
    

    def _one_step(self, nt, sources_xz, 
        prev_wavefield, cur_wavefield,          
        cur_psi, cur_phi,
        model_padded
        ):
        nabla_u = self.laplacian(cur_wavefield)
        phi_x = self.first_x_deriv(cur_phi)
        psi_z = self.first_z_deriv(cur_psi)
        ux = self.first_x_deriv(cur_wavefield)
        uz = self.first_z_deriv(cur_wavefield)

        next_wavefield = self.dt**2 * model_padded**2 * (nabla_u + phi_x + psi_z) \
                    - self.dt**2 * self.sigma_x * self.sigma_z * cur_wavefield\
                    + (self.dt/2) * (self.sigma_x + self.sigma_z) * prev_wavefield\
                    + 2 * cur_wavefield - prev_wavefield
        
        next_wavefield = next_wavefield / (1 + (self.dt/2) * (self.sigma_x + self.sigma_z))
    
        # add source 
        sx = sources_xz[:, 1] + self.total_pad
        sz = sources_xz[:, 0] + self.total_pad 
        num_shots = len(sources_xz)
        next_wavefield[range(num_shots), sz, sx] += self.dt**2 * self.source_time[nt]

        next_phi = -self.dt * self.sigma_x * cur_phi \
            + self.dt * (self.sigma_z - self.sigma_x) * ux + cur_phi
        
        next_psi = -self.dt * self.sigma_z * cur_psi \
            + self.dt * (self.sigma_x - self.sigma_z) * uz + cur_psi

        return next_wavefield, next_phi, next_psi
        

    def step(self, sources_xz):
        # Model 
        assert torch.all(self.model > 0) 
        assert not torch.any(torch.isnan(self.model))

        model_padded = self._set_model()

        # wavefield
        num_shots = len(sources_xz)
        prev_wavefield = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
        cur_wavefield = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)

        # auxiliary function
        cur_psi = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
        cur_phi = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device) 

        num_receivers = len(self.receivers_xz)
        seismogram = torch.zeros((num_shots, num_receivers, len(self.source_time)), dtype=torch.float32, device=self.device)

        for nt in range(self.nt):
            next_wavefield, next_phi, next_psi = self._one_step(nt, sources_xz, 
                                                                prev_wavefield, cur_wavefield,          
                                                                cur_psi, cur_phi,
                                                                model_padded)  
            actual_wavefield = next_wavefield[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
            seismogram[:, :, nt] = actual_wavefield[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]

            cur_wavefield, prev_wavefield = next_wavefield, cur_wavefield 
            cur_phi, cur_psi = next_phi, next_psi 
        
        return seismogram






