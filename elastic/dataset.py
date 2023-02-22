"""
Utils about data set for full waveform training, i.e. sythetic data 
generate by true undergroud physical parameters of elastic wave equation

@author: zchen
@date: 2023-02-20
"""

import torch
from solver import Solver


class SytheticData(torch.utils.data.Dataset):
    def __init__(self, rho, vp, vs, 
                h, dt, nt, peak_time, dominant_freq,
                sources_xz, receivers_xz, 
                pml_width=10, pad_width=10):
        super().__init__()
        self.sources_xz = sources_xz 
        sol = Solver(
            rho, vp, vs, h, dt, nt, 
            peak_time, dominant_freq, receivers_xz,
            pml_width, pad_width
            )
        self.data_vx, self.data_vy = sol.step(sources_xz)
        if self.data_vx.is_cuda:
            self.data_vx = self.data_vx.cpu()
            self.data_vy = self.data_vy.cpu()

    def __len__(self):
        return len(self.sources_xz)

    def __getitem__(self, id):
        return self.sources_xz[id], self.data_vx[id], self.data_vy[id]




    
