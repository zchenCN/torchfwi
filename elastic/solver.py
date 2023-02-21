"""
Staggered grid Finite difference solver for 2d elastic wave equations
with perfectly matched layer. Here the code is implemented by PyTorch,
for auto differentiation.

For more details about the PML and finite difference method,
please refernce this paper: https://hal.inria.fr/inria-00073219/file/RR-3471.pdf

@author: zchen
@date: 2023-02-16
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
    return w * 1e4


class Solver:
    def __init__(self, rho, vp, vs, 
                h, dt, nt, peak_time, dominant_freq,
                receivers_xz, 
                pml_width=10, pad_width=10):
        
        assert rho.device == vp.device == vs.device 
        assert rho.shape == vp.shape == vs.shape
        assert rho.dtype == vp.dtype == vs.dtype == torch.float32

        # Device 
        self.device = rho.device
        
        # Mesh
        self.nptz, self.nptx = rho.shape[0], rho.shape[1]
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
        self.rho = rho 
        self.vp = vp 
        self.vs = vs

        # Dampling factors
        self._calc_dampling_factors()

    def _set_model(self):
        """ Add padding to model parameters and interpolation them for
        stagerred grid finite difference method
        """
        # Padding
        pad = nn.ReplicationPad2d((self.total_pad,) * 4)
        rho_padded = pad(self.rho[None, :]).squeeze()
        vp_padded = pad(self.vp[None, :]).squeeze()
        vs_padded = pad(self.vs[None, :]).squeeze()

        mu_padded = rho_padded * vs_padded**2
        ld_padded = rho_padded * vp_padded**2 - 2 * mu_padded

        # Interpolate
        mu_padded_ihj = (mu_padded[:, :-1] + mu_padded[:, 1:]) / 2
        mu_padded_ijh = (mu_padded[:-1, :] + mu_padded[1:, :]) / 2
        ld_padded_ihj = (ld_padded[:, :-1] + ld_padded[:, 1:]) / 2
        rho_padded_ihj = (rho_padded[:, :-1] + rho_padded[:, 1:]) / 2
        rho_padded_ihjh = (rho_padded_ihj[:-1, :] + rho_padded_ihj[1:, :]) / 2

        return rho_padded, rho_padded_ihjh, ld_padded_ihj, mu_padded_ihj, mu_padded_ijh


    def _calc_dampling_factors(self):
        # Dampling factor
        profile = 40.0 + 60.0 * np.arange(self.pml_width, dtype=np.float32) 
        profile_h = 40.0 + 60.0 * (np.arange(self.pml_width, dtype=np.float32) - 0.5) # profile at half grid 

        d_x = np.zeros(self.nptx_padded, np.float32)
        d_x[self.total_pad-1:self.pad_width-1:-1] = profile 
        d_x[-self.total_pad:-self.pad_width] = profile
        d_x[:self.pad_width] = d_x[self.pad_width]
        d_x[-self.pad_width:] = d_x[-self.pad_width-1]

        d_xh = np.zeros(self.nptx_padded - 1, np.float32)
        d_xh[self.total_pad-1:self.pad_width-1:-1] = profile_h 
        d_xh[-self.total_pad:-self.pad_width] = profile_h 
        d_xh[:self.pad_width] = d_x[self.pad_width]
        d_xh[-self.pad_width:] = d_x[-self.pad_width-1]

        d_y = np.zeros(self.nptz_padded, np.float32)
        d_y[self.total_pad-1:self.pad_width-1:-1] = profile 
        d_y[-self.total_pad:-self.pad_width] = profile
        d_y[:self.pad_width] = d_y[self.pad_width]
        d_y[-self.pad_width:] = d_y[-self.pad_width-1]

        d_yh = np.zeros(self.nptz_padded - 1, np.float32)
        d_yh[self.total_pad-1:self.pad_width-1:-1] = profile_h
        d_yh[-self.total_pad:-self.pad_width] = profile_h 
        d_yh[:self.pad_width] = d_y[self.pad_width]
        d_yh[-self.pad_width:] = d_y[-self.pad_width-1]

        dx_ij = np.tile(d_x, (self.nptz_padded, 1))
        dx_ijh = np.tile(d_x, (self.nptz_padded-1, 1))
        dx_ihj = np.tile(d_xh, (self.nptz_padded, 1))
        dx_ihjh = np.tile(d_xh, (self.nptz_padded-1, 1))
        dy_ij = np.tile(d_y.reshape(-1, 1), (1, self.nptx_padded))
        dy_ihj = np.tile(d_y.reshape(-1, 1), (1, self.nptx_padded-1))
        dy_ijh = np.tile(d_yh.reshape(-1, 1), (1, self.nptx_padded))
        dy_ihjh = np.tile(d_yh.reshape(-1, 1), (1, self.nptx_padded-1))

        self.dx_ij = torch.tensor(dx_ij, dtype=torch.float32, device=self.device)
        self.dx_ijh = torch.tensor(dx_ijh, dtype=torch.float32, device=self.device)
        self.dx_ihj = torch.tensor(dx_ihj, dtype=torch.float32, device=self.device)
        self.dx_ihjh = torch.tensor(dx_ihjh, dtype=torch.float32, device=self.device)
        self.dy_ij = torch.tensor(dy_ij, dtype=torch.float32, device=self.device)
        self.dy_ijh = torch.tensor(dy_ijh, dtype=torch.float32, device=self.device)
        self.dy_ihj = torch.tensor(dy_ihj, dtype=torch.float32, device=self.device)
        self.dy_ihjh = torch.tensor(dy_ihjh, dtype=torch.float32, device=self.device)

    def _sxx_first_x_deriv(self, sxx):
        sxx_x = (sxx[:, :, 1:] - sxx[:, :, :-1]) / self.h
        sxx_x = F.pad(sxx_x, (1, 1), "constant", 0.0)
        return sxx_x 

    def _sxy_first_x_deriv(self, sxy):
        sxy_x = (sxy[:, :, 1:] - sxy[:, :, :-1]) / self.h 
        return sxy_x

    def _sxy_first_y_deriv(self, sxy):
        sxy_y = (sxy[:, 1:, :] - sxy[:, :-1, :]) / self.h 
        sxy_y = F.pad(sxy_y, (0, 0, 1, 1), "constant", 0.0)
        return sxy_y 

    def _syy_first_y_deriv(self, syy):
        syy_y = (syy[:, 1:, :] - syy[:, :-1, :]) / self.h 
        return syy_y 

    def _vx_first_x_deriv(self, vx):
        vx_x = (vx[:, :, 1:] - vx[:, :, :-1]) / self.h 
        return vx_x 

    def _vx_first_y_deriv(self, vx):
        vx_y = (vx[:, 1:, :] - vx[:, :-1, :]) / self.h 
        return vx_y

    def _vy_first_x_deriv(self, vy):
        vy_x = (vy[:, :, 1:] - vy[:, :, :-1]) / self.h
        vy_x = F.pad(vy_x, (1, 1), "constant", 0.0)
        return vy_x 

    def _vy_first_y_deriv(self, vy):
        vy_y = (vy[:, 1:, :] - vy[:, :-1, :]) / self.h
        vy_y = F.pad(vy_y, (0, 0, 1, 1), "constant", 0.0)
        return vy_y 

    def _one_step(
        self, nt, sources_xz,
        cur_vx_x, cur_vx_y,
        cur_vy_x, cur_vy_y,
        cur_sxx_x, cur_sxx_y,
        cur_sxy_x, cur_sxy_y,
        cur_syy_x, cur_syy_y,
        rho_padded, rho_padded_ihjh,
        ld_padded_ihj,
        mu_padded_ihj, mu_padded_ijh,
        ):
        
        num_shots = len(sources_xz)

        cur_vx = cur_vx_x + cur_vx_y 
        cur_vy = cur_vy_x + cur_vy_y 


        vx_x = self._vx_first_x_deriv(cur_vx)
        vx_y = self._vx_first_y_deriv(cur_vx)
        vy_x = self._vy_first_x_deriv(cur_vy)
        vy_y = self._vy_first_y_deriv(cur_vy)

        next_sxx_x = (
            (ld_padded_ihj + 2 * mu_padded_ihj) * self.dt * vx_x 
            + cur_sxx_x 
            - self.dt * self.dx_ihj * cur_sxx_x / 2
        )
        next_sxx_x /= (1 + self.dt * self.dx_ihj / 2)

        next_sxx_y = (
            ld_padded_ihj * self.dt * vy_y 
            + cur_sxx_y
            - self.dt * self.dy_ihj * cur_sxx_y / 2
        )
        next_sxx_y /= (1 + self.dt * self.dy_ihj / 2)

        next_syy_x = (
            ld_padded_ihj * self.dt * vx_x 
            + cur_syy_x
            - self.dt * self.dx_ihj * cur_syy_x / 2
        )
        next_syy_x /= (1 + self.dx_ihj * self.dt / 2)

        next_syy_y = (
            (ld_padded_ihj + 2 * mu_padded_ihj) * self.dt * vy_y 
            + cur_syy_y 
            - self.dt * self.dy_ihj * cur_syy_y / 2
        )
        next_syy_y /= (1 + self.dy_ihj * self.dt / 2)

        next_sxy_x = (
            mu_padded_ijh * self.dt * vy_x 
            + cur_sxy_x 
            - self.dt * self.dx_ijh * cur_sxy_x / 2
        )
        next_sxy_x /= (1 + self.dt * self.dx_ijh / 2)

        next_sxy_y = (
            mu_padded_ijh * self.dt * vx_y 
            + cur_sxy_y
            - self.dt * self.dy_ijh * cur_sxy_y / 2
        )
        next_sxy_y /= (1 + self.dy_ijh * self.dt / 2)

        next_sxx = next_sxx_x + next_sxx_y 
        next_sxy = next_sxy_x + next_sxy_y
        next_syy = next_syy_x + next_syy_y

        sxx_x = self._sxx_first_x_deriv(next_sxx)
        sxy_y = self._sxy_first_y_deriv(next_sxy)
        sxy_x = self._sxy_first_x_deriv(next_sxy)
        syy_y = self._syy_first_y_deriv(next_syy)

        next_vx_x = (
            self.dt * sxx_x / rho_padded
            + cur_vx_x 
            - self.dx_ij * self.dt * cur_vx_x / 2
        ) 
        next_vx_x /= (1 + self.dt * self.dx_ij / 2)

        next_vx_y = ( 
            self.dt * sxy_y / rho_padded
            + cur_vx_y 
            - self.dy_ij * self.dt * cur_vx_y / 2
        )
        next_vx_y /= (1 + self.dt * self.dy_ij / 2)

        next_vy_x = ( 
            self.dt *  sxy_x / rho_padded_ihjh
            + cur_vy_x 
            - self.dt * self.dx_ihjh * cur_vy_x / 2
        )
        next_vy_x /= (1 + self.dt * self.dx_ihjh / 2)

        next_vy_y = (
            self.dt * syy_y / rho_padded_ihjh
            + cur_vy_y 
            - self.dt * self.dy_ihjh * cur_vy_y / 2
        )
        next_vy_y /= (1 + self.dt * self.dy_ihjh / 2)
                

        # Add source 
        sx = sources_xz[:, 1] + self.total_pad
        sz = sources_xz[:, 0] + self.total_pad 
        next_vx_x[range(num_shots), sz, sx] += self.dt * self.source_time[nt]

        next_vx = next_vx_x + next_vx_y 
        next_vy = next_vy_x + next_vy_y 

        return (
            next_vx, next_vy, 
            next_vx_x, next_vx_y, 
            next_vy_x, next_vy_y,
            next_sxx_x, next_sxx_y,
            next_sxy_x, next_sxy_y,
            next_syy_x, next_syy_y       
            )

    def step(self, sources_xz):
        # Model 
        assert torch.all(self.rho > 0) and torch.all(self.vp > 0) and torch.all(self.vs > 0)
        assert not torch.any(torch.isnan(self.rho)) and not torch.any(torch.isnan(self.vp)) and not torch.any(torch.isnan(self.vs))
        rho_padded, rho_padded_ihjh, ld_padded_ihj, mu_padded_ihj, mu_padded_ijh = self._set_model()

        # Stress and velocity
        num_shots = len(sources_xz)
        cur_vx_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
        cur_vx_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
        cur_vy_x = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded-1), dtype=torch.float32, device=self.device)
        cur_vy_y = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded-1), dtype=torch.float32, device=self.device)
        cur_sxx_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)
        cur_sxx_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)
        cur_syy_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)
        cur_syy_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)
        cur_sxy_x = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded), dtype=torch.float32, device=self.device)
        cur_sxy_y = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded), dtype=torch.float32, device=self.device)

        num_receivers = len(self.receivers_xz)
        seismogram_vx = torch.zeros((num_shots, num_receivers, len(self.source_time)), dtype=torch.float32, device=self.device)
        seismogram_vy = torch.zeros((num_shots, num_receivers, len(self.source_time)), dtype=torch.float32, device=self.device)

        for nt in range(self.nt):
            state = self._one_step(
                nt, sources_xz,
                cur_vx_x, cur_vx_y,
                cur_vy_x, cur_vy_y,
                cur_sxx_x, cur_sxx_y,
                cur_sxy_x, cur_sxy_y,
                cur_syy_x, cur_syy_y,
                rho_padded, rho_padded_ihjh,
                ld_padded_ihj,
                mu_padded_ihj, mu_padded_ijh,
            )
            next_vx, next_vy = state[0], state[1]
            actual_vx = next_vx[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
            actual_vy = next_vy[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
             
            if self.receivers_xz is not None:
                seismogram_vx[:, :, nt] = actual_vx[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
                seismogram_vy[:, :, nt] = actual_vy[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]

            cur_vx_x, cur_vx_y = state[2], state[3]
            cur_vy_x, cur_vy_y = state[4], state[5]
            cur_sxx_x, cur_sxx_y = state[6], state[7]
            cur_sxy_x, cur_sxy_y = state[8], state[9]
            cur_syy_x, cur_syy_y = state[10], state[11]

        return seismogram_vx, seismogram_vy

