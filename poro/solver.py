"""
Staggered grid Finite difference solver for 2d poro-elastic wave equations
with perfectly matched layer. Here the code is implemented by PyTorch,
for auto differentiation.


@author: zchen
@date: 2023-03-24
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
    return w 


class Solver:
    def __init__(self, ld, mu, ks, kf, 
                 rhos, rhof, phi, a, 
                 h, dt, nt, peak_time, dominant_freq,
                 receivers_xz,
                 pml_width=10, pad_width=10
                 ):
        
        assert ld.device == mu.device == ks.device == kf.device 
        assert ld.shape == mu.shape == ks.shape == kf.shape 
        assert ld.dtype == mu.dtype == ks.dtype == kf.dtype == torch.float32 

        # Device 
        self.device = ld.device 

        # Mesh
        self.nptz, self.nptx = ld.shape[0], ld.shape[1]
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
        self.ld, self.mu = ld, mu 
        self.ks, self.kf = ks, kf 
        self.rhos, self.rhof = rhos, rhof 
        self.phi = phi 
        self.a = a 

        # Dampling factors
        self._calc_dampling_factors()


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

    def _set_model(self):
        """ Add padding to model parameters and interpolation them for
        stagerred grid finite difference method
        """
        # Padding
        pad = nn.ReplicationPad2d((self.total_pad,) * 4)
        ld_padded = pad(self.ld[None, :]).squeeze()
        mu_padded = pad(self.mu[None, :]).squeeze()
        ks_padded = pad(self.ks[None, :]).squeeze()
        kf_padded = pad(self.kf[None, :]).squeeze()
        rhos_padded = pad((self.rhos[None, :])).squeeze()
        rhof_padded = pad((self.rhof[None, :])).squeeze()

        # Data on (i, j)
        m_padded = self.a * rhof_padded / self.phi 
        rho_padded = self.phi * rhof_padded + (1 - self.phi) * rhos_padded
        kb_padded = ld_padded + 2 * mu_padded / 3
        alpha_padded = 1 - kb_padded / ks_padded
        M_padded = 1 / (self.phi / kf_padded + (alpha_padded - self.phi) / ks_padded)

        # Interpolate data on (i+1/2, j)
        m_padded_ihj = (m_padded[:, 1:] + m_padded[:, :-1]) / 2
        rhof_padded_ihj = (rhof_padded[:, 1:] + rhof_padded[:, :-1]) / 2
        rho_padded_ihj = (rho_padded[:, 1:] + rho_padded[:, :-1]) / 2

        # Interpolate data on (i, j+1/2)
        m_padded_ijh = (m_padded[1:, :] + m_padded[:-1, :]) / 2
        rhof_padded_ijh = (rhof_padded[1:, :] + rhof_padded[:-1, :]) / 2
        rho_padded_ijh = (rho_padded[1:, :] + rho_padded[:-1, :]) / 2

        # # Interpolate data on (i+1/2, j+1/2)
        mu_padded_ihj = (mu_padded[:, 1:] + mu_padded[:, :-1]) / 2
        mu_padded_ihjh = (mu_padded_ihj[1:, :] + mu_padded_ihj[:-1, :]) / 2

        return (ld_padded, mu_padded, alpha_padded, M_padded,
                m_padded_ihj, rhof_padded_ihj, rho_padded_ihj,
                m_padded_ijh, rhof_padded_ijh, rho_padded_ijh,
                mu_padded_ihjh
        )

    def _first_x_deriv_ij(self, f):
        """First derivative of x for data on (i, j), fx lay on (i+1/2, j)
        """
        fx = (f[:, :, 1:] - f[:, :, :-1]) / self.h 
        return fx 
    
    def _first_x_deriv_ihj(self, f):
        """First derivative of x for data on (i+1/2, j), fx lay on (i, j)
        """
        fx = (f[:, :, 1:] - f[:, :, :-1]) / self.h 
        fx = F.pad(fx, (1, 1), "constant", 0.0)
        return fx 
    
    def _first_x_deriv_ijh(self, f):
        """First derivative of x for data on (i, j+1/2), fx lay on (i+1/2, j+1/2)
        """
        fx = (f[:, :, 1:] - f[:, :, :-1]) / self.h 
        return fx 
    
    def _first_x_deriv_ihjh(self, f):
        """First derivative of x for data on (i+1/2, j+1/2), fx lay on (i, j+1/2)
        """
        fx = (f[:, :, 1:] - f[:, :, :-1]) / self.h 
        fx = F.pad(fx, (1, 1), "constant", 0.0)
        return fx 
    
    def _first_y_deriv_ij(self, f):
        """First derivative of y for data on (i, j), fy lay on (i, j+1/2)
        """
        fy = (f[:, 1:, :] - f[:, :-1, :]) / self.h 
        return fy

    def _first_y_deriv_ihj(self, f):
        """First derivative of y for data on (i+1/2, j), fy lay on (i+1/2, j+1/2)
        """
        fy = (f[:, 1:, :] - f[:, :-1, :]) / self.h 
        return fy
    
    def _first_y_deriv_ijh(self, f):
        """First derivative of y for data on (i, j+1/2), fy lay on (i, j)
        """
        fy = (f[:, 1:, :] - f[:, :-1, :]) / self.h 
        fy = F.pad(fy, (0, 0, 1, 1), "constant", 0.0)
        return fy

    def _first_y_deriv_ihjh(self, f):
        """First derivative of y for data on (i+1/2, j+1/2), fy lay on (i+1/2, j)
        """
        fy = (f[:, 1:, :] - f[:, :-1, :]) / self.h 
        fy = F.pad(fy, (0, 0, 1, 1), "constant", 0.0)
        return fy


    def _one_step(self, nt, sources_xz,
                cur_vsx_x, cur_vsx_y,
                cur_vfx_x, cur_vfx_y,
                cur_vsy_x, cur_vsy_y,
                cur_vfy_x, cur_vfy_y,
                cur_exx_x, cur_exx_y,
                cur_eyy_x, cur_eyy_y,
                cur_e_x, cur_e_y,
                cur_exy_x, cur_exy_y,
                ld_padded, mu_padded, alpha_padded, M_padded,
                m_padded_ihj, rhof_padded_ihj, rho_padded_ihj,
                m_padded_ijh, rhof_padded_ijh, rho_padded_ijh,
                mu_padded_ihjh
    ):
        
        cur_exx = cur_exx_x + cur_exx_y 
        cur_eyy = cur_eyy_x + cur_eyy_y 
        cur_exy = cur_exy_x + cur_exy_y 
        cur_e = cur_e_x + cur_e_y 

        # spatial derivativies 
        dx_exx = self._first_x_deriv_ij(cur_exx)
        dy_eyy = self._first_y_deriv_ij(cur_eyy)
        dx_e = self._first_x_deriv_ij(cur_e)
        dy_e = self._first_y_deriv_ij(cur_e)
        dx_exy = self._first_x_deriv_ihjh (cur_exy)
        dy_exy = self._first_y_deriv_ihjh(cur_exy)

        # update vsx, vsy, vfx, vfy 
        next_vsx_x = (
            (m_padded_ihj * dx_exx - rhof_padded_ihj * dx_e) / (m_padded_ihj * rho_padded_ihj - rhof_padded_ihj**2)
             + (1 / self.dt) * cur_vsx_x - self.dx_ihj * cur_vsx_x / 2
        )
        next_vsx_x /= ((1 / self.dt) + self.dx_ihj / 2)

        next_vsx_y = (
            (m_padded_ihj * dy_exy) / (m_padded_ihj * rho_padded_ihj - rhof_padded_ihj**2)
             + (1 / self.dt) * cur_vsx_y - self.dy_ihj * cur_vsx_y / 2
        )
        next_vsx_y /= ((1 / self.dt) + self.dy_ihj / 2)

        next_vsy_x = (
            (m_padded_ijh * dx_exy) / (m_padded_ijh * rho_padded_ijh - rhof_padded_ijh**2)
             + (1 / self.dt) * cur_vsy_x - self.dx_ijh * cur_vsy_x / 2
        )
        next_vsy_x /= ((1 / self.dt) + self.dx_ijh / 2)

        next_vsy_y = (
            (m_padded_ijh * dy_eyy - rhof_padded_ijh * dy_e) / (m_padded_ijh * rho_padded_ijh - rhof_padded_ijh**2)
             + (1 / self.dt) * cur_vsy_y - self.dy_ijh * cur_vsy_y / 2
        )
        next_vsy_y /= ((1 / self.dt) + self.dy_ijh / 2)

        next_vfx_x = (
            (-rhof_padded_ihj * dx_exx + rho_padded_ihj * dx_e) / (m_padded_ihj * rho_padded_ihj - rhof_padded_ihj**2)
             + (1 / self.dt) * cur_vfx_x - self.dx_ihj * cur_vfx_x / 2
        )
        next_vfx_x /= ((1 / self.dt) + self.dx_ihj / 2)

        next_vfx_y = (
            (-rhof_padded_ihj * dy_exy) / (m_padded_ihj * rho_padded_ihj - rhof_padded_ihj**2)
             + (1 / self.dt) * cur_vfx_y - self.dy_ihj * cur_vfx_y / 2
        )
        next_vfx_y /= ((1 / self.dt) + self.dy_ihj / 2)

        next_vfy_x = (
            (-rhof_padded_ijh * dx_exy) / (m_padded_ijh * rho_padded_ijh - rhof_padded_ijh**2)
             + (1 / self.dt) * cur_vfy_x - self.dx_ijh * cur_vfy_x / 2
        )
        next_vfy_x /= ((1 / self.dt) + self.dx_ijh / 2)

        next_vfy_y = (
            (-rhof_padded_ijh * dy_eyy + rho_padded_ijh * dy_e) / (m_padded_ijh * rho_padded_ijh - rhof_padded_ijh**2)
             + (1 / self.dt) * cur_vfy_y - self.dy_ijh * cur_vfy_y / 2
        )
        next_vfy_y /= ((1 / self.dt) + self.dy_ijh / 2)

        # Add source 
        num_shots = len(sources_xz)
        sx = sources_xz[:, 1] + self.total_pad
        sz = sources_xz[:, 0] + self.total_pad 
        next_vsx_x[range(num_shots), sz, sx] += self.dt * self.source_time[nt]
        next_vsx_y[range(num_shots), sz, sx] += self.dt * self.source_time[nt]

        next_vsx = next_vsx_x + next_vsx_y 
        next_vsy = next_vsy_x + next_vsy_y
        next_vfx = next_vfx_x + next_vfx_y 
        next_vfy = next_vfy_x + next_vfy_y 

        dx_vsx = self._first_x_deriv_ihj(next_vsx)
        dy_vsx = self._first_y_deriv_ihj(next_vsx)
        dx_vsy = self._first_x_deriv_ijh(next_vsy)
        dy_vsy = self._first_y_deriv_ijh(next_vsy)
        dx_vfx = self._first_x_deriv_ihj(next_vfx)
        dy_vfy = self._first_y_deriv_ijh(next_vfy)

        # Update exx, eyy, exy, e
        next_exx_x = (
            (ld_padded + 2 * mu_padded) * dx_vsx + alpha_padded * M_padded * dx_vfx 
            + (1 / self.dt) * cur_exx_x - self.dx_ij * cur_exx_x / 2
        )
        next_exx_x /= ((1 / self.dt) + self.dx_ij / 2)

        next_exx_y = (
            ld_padded * dy_vsy + alpha_padded * M_padded * dy_vfy 
            + (1 / self.dt) * cur_exx_y - self.dy_ij * cur_exx_y / 2
        )
        next_exx_y /= ((1 / self.dt) + self.dy_ij / 2)

        next_exy_x = (
            mu_padded_ihjh * dx_vsy 
            + (1 / self.dt) * cur_exy_x - self.dx_ihjh * cur_exy_x / 2
        )
        next_exy_x /= ((1 / self.dt) + self.dx_ihjh / 2)

        next_exy_y = (
            mu_padded_ihjh * dy_vsx
            + (1 / self.dt) * cur_exy_y - self.dy_ihjh * cur_exy_y / 2
        )
        next_exy_y /= ((1 / self.dt) + self.dy_ihjh / 2)

        next_eyy_x = (
            ld_padded * dx_vsx + alpha_padded * M_padded * dx_vfx 
            + (1 / self.dt) * cur_eyy_x - self.dx_ij * cur_eyy_x / 2
        )
        next_eyy_x /= ((1 / self.dt) + self.dx_ij / 2)

        next_eyy_y = (
            (ld_padded + 2 * mu_padded) * dy_vsy + alpha_padded * M_padded * dy_vfy 
            + (1 / self.dt) * cur_eyy_y - self.dy_ij * cur_eyy_y / 2
        )
        next_eyy_y /= ((1 / self.dt) + self.dy_ij / 2)

        next_e_x = (
            alpha_padded * M_padded * dx_vsx + M_padded * dx_vfx 
            + (1 / self.dt) * cur_e_x - self.dx_ij * cur_e_x / 2
        )
        next_e_x /= ((1 / self.dt) + self.dx_ij / 2)

        next_e_y = (
            alpha_padded * M_padded * dy_vsy + M_padded * dy_vfy 
            + (1 / self.dt) * cur_e_y - self.dy_ij * cur_e_y / 2
        )
        next_e_y /= ((1 / self.dt) + self.dy_ij / 2)

        return (
            next_vsx, next_vsy,
            next_vfx, next_vfy,
            next_vsx_x, next_vsx_y,
            next_vfx_x, next_vfx_y,
            next_vsy_x, next_vsy_y,
            next_vfy_x, next_vfy_y,
            next_exx_x, next_exx_y,
            next_exy_x, next_exy_y,
            next_eyy_x, next_eyy_y,
            next_e_x, next_e_y,
        )

    def step(self, sources_xz):
            # Model 
            assert torch.all(self.ld > 0) and torch.all(self.mu > 0) and torch.all(self.ks > 0) and torch.all(self.kf > 0)
            assert not torch.any(torch.isnan(self.ld)) and not torch.any(torch.isnan(self.mu)) and not torch.any(torch.isnan(self.ks)) and not torch.any(torch.isnan(self.kf))
            (ld_padded, mu_padded, alpha_padded, M_padded,
            m_padded_ihj, rhof_padded_ihj, rho_padded_ihj,
            m_padded_ijh, rhof_padded_ijh, rho_padded_ijh,
            mu_padded_ihjh
            ) = self._set_model()

            num_shots = len(sources_xz)
            # Data on (i, j)
            cur_e_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_e_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_exx_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_exx_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_eyy_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_eyy_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded), dtype=torch.float32, device=self.device)

            # Data on (i+1/2, j)
            cur_vsx_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)
            cur_vsx_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)
            cur_vfx_x = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)
            cur_vfx_y = torch.zeros((num_shots, self.nptz_padded, self.nptx_padded-1), dtype=torch.float32, device=self.device)

            # Data on (i, j+1/2)
            cur_vsy_x = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_vsy_y = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_vfy_x = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded), dtype=torch.float32, device=self.device)
            cur_vfy_y = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded), dtype=torch.float32, device=self.device)

            # Data on (i+1/2, j+1/2)
            cur_exy_x = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded-1), dtype=torch.float32, device=self.device)
            cur_exy_y = torch.zeros((num_shots, self.nptz_padded-1, self.nptx_padded-1), dtype=torch.float32, device=self.device)

            num_receivers = len(self.receivers_xz)
            seis_vsx = torch.zeros((num_shots, num_receivers, len(self.source_time)), dtype=torch.float32, device=self.device)
            seis_vsy = torch.zeros((num_shots, num_receivers, len(self.source_time)), dtype=torch.float32, device=self.device)
            seis_vfx = torch.zeros((num_shots, num_receivers, len(self.source_time)), dtype=torch.float32, device=self.device)
            seis_vfy = torch.zeros((num_shots, num_receivers, len(self.source_time)), dtype=torch.float32, device=self.device)

            # propagation 
            for nt in range(self.nt):
                state = self._one_step(
                    nt, sources_xz, 
                    cur_vsx_x, cur_vsx_y,
                    cur_vfx_x, cur_vfx_y,
                    cur_vsy_x, cur_vsy_y,
                    cur_vfy_x, cur_vfy_y,
                    cur_exx_x, cur_exx_y,
                    cur_eyy_x, cur_eyy_y,
                    cur_e_x, cur_e_y,
                    cur_exy_x, cur_exy_y,
                    ld_padded, mu_padded, alpha_padded, M_padded,
                    m_padded_ihj, rhof_padded_ihj, rho_padded_ihj,
                    m_padded_ijh, rhof_padded_ijh, rho_padded_ijh,
                    mu_padded_ihjh
                )

                next_vsx, next_vsy, next_vfx, next_vfy = state[0], state[1], state[2], state[3]
                actual_vsx = next_vsx[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
                actual_vsy = next_vsy[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
                actual_vfx = next_vfx[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]
                actual_vfy = next_vfy[:, self.total_pad:-self.total_pad, self.total_pad:-self.total_pad]

                # Seismogram 
                if self.receivers_xz is not None:
                    seis_vsx[:, :, nt] = actual_vsx[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
                    seis_vsy[:, :, nt] = actual_vsy[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
                    seis_vfx[:, :, nt] = actual_vfx[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]
                    seis_vfy[:, :, nt] = actual_vfy[:, self.receivers_xz[:, 0], self.receivers_xz[:, 1]]

                # Update current state 
                cur_vsx_x, cur_vsx_y = state[4], state[5]
                cur_vfx_x, cur_vfx_y = state[6], state[7]
                cur_vsy_x, cur_vsy_y = state[8], state[9]
                cur_vfy_x, cur_vfy_y = state[10], state[11]
                cur_exx_x, cur_exx_y = state[12], state[13]
                cur_exy_x, cur_exy_y = state[14], state[15]
                cur_eyy_x, cur_eyy_y = state[16], state[17]
                cur_e_x, cur_e_y = state[18], state[19]

            return seis_vsx, seis_vsy, seis_vfx, seis_vfy


