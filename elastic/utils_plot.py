"""
Utilities for plotting underground physical model parameters

@author: zchen
@date: 2023-02-21
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms

def relative_error(true, inversion):
    error = np.linalg.norm(true - inversion)**2 / np.linalg.norm(true) **2
    error = np.sqrt(error)
    return error


def plot_model(model, name, h, path=None):
    nptz, nptx = model.shape

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    im = ax.imshow(model)
    
    # setting labels and ticks
    ax.set_xlabel('X (km)', fontsize=16)
    ax.set_ylabel('Z (km)', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(0, nptx, 30))
    ax.set_xticklabels(np.arange(0, nptx, 30)*h, fontsize=16)
    ax.set_yticks(np.arange(0, nptz, 20))
    ax.set_yticklabels(np.arange(0, nptz, 20)*h, fontsize=16)
    
    # setting colorbar
    ax_pos = ax.get_position()
    pad = 0.02
    width = 0.02
    cax_pos = mtransforms.Bbox.from_extents(
        ax_pos.x1 + pad,
        ax_pos.y0, #+ 0.085,
        ax_pos.x1 + pad + width,
        ax_pos.y1 #- 0.085
    )
    cax = ax.figure.add_axes(cax_pos)
    cbar = fig.colorbar(im, cax=cax, 
                       orientation='vertical')
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('km/s', fontsize=16) 

    # save figure or plot it
    if not path:
        plt.show()
    else:
        save_path = path + name + '.png'
        plt.savefig(save_path, bbox_inches = 'tight')

        
def plot_section(true, initial, inversion, h, id=None, name=None, path=None):
    assert true.shape == initial.shape == inversion.shape
    nptz, nptx = true.shape
    
    if id is None:
        id = nptx // 2

    plt.figure()
    plt.xlabel('Z (km)')
    plt.ylabel('Velocity (km/s)')
    error_initial = relative_error(initial[:, id], true[:, id])
    error_inversion = relative_error(inversion[:, id], true[:, id])
    print(f'The relative error of initial model is {error_initial*100:.2f}%')
    print(f'The relative error of inversion result is {error_inversion*100:.2f}%')
    plt.plot(true[:, id], 'r--', linewidth=1.0, label='True')
    plt.plot(initial[:, id], 'g-.', linewidth=1.0, label='Initial')
    plt.plot(inversion[:, id], 'b-', linewidth=1.0, label='Inversion')

    ax = plt.gca()
    ax.set_xticks(np.arange(0, nptz, 10))
    # ax.set_xticklabels(np.arange(0, nptz, 10)*h)
    ax.set_xticklabels([f'{ts:.1f}' for ts in np.arange(0, nptz, 10)*h])
    plt.legend()
    
    # save figure or plot it
    if not path:
        plt.show()
    else:
        save_path = path + name + '.png'
        plt.savefig(save_path, bbox_inches = 'tight')

def plot_seismogram(data, dt, h):
    nt, num_receivers = data.shape
    plt.imshow(data, aspect='auto', cmap='gray')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel('x (km)', fontsize=12)
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('t (s)', fontsize=12)
    ax.set_yticks(np.arange(0, nt, 300))
    ax.set_yticklabels(np.arange(0, nt, 300)*dt)
    ax.set_xticks(np.arange(0, num_receivers, 30))
    ax.set_xticklabels(np.arange(0, num_receivers, 30)*h)
    
