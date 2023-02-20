"""
Utils about data set for full waveform training, i.e. sythetic data 
generate by true undergroud physical parameters of elastic wave equation

@author: zchen
@date: 2023-02-20
"""

import torch


class SytheticData(torch.utils.data.Dataset):
    def __init__(self, solver, sources_xz):
        super().__init__()
        self.sources_xz = sources_xz 
        self.data_vx, self.data_vy = solver.step(sources_xz)

    def __len__(self):
        return len(self.sources_xz)

    def __getitem__(self, id):
        return self.sources_xz[id], self.data_vx[id], self.data_vy[id]




    
