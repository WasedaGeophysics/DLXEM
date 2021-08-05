import numpy as np
import matplotlib.pyplot as plt

def thicks_maker(init_thick, last_depth, nlayer, mode):
    if mode == 'log':
        depth = np.logspace(np.log10(init_thick), np.log10(last_depth), nlayer-1)
        depth = np.append([0], depth)
        thicks = []
        for i in range(nlayer-1):
            thicks.append(depth[i+1]-depth[i])

        fig = plt.figure(figsize=(25,1))
        ax = fig.add_subplot(111)
        for i in range(nlayer):
            ax.plot([depth[i], depth[i]], [0,1], c='k')

        ax.set_xscale('log')

    elif mode == 'linear':
        depth = np.linspace(init_thick, last_depth, nlayer-1)
        if init_thick == 0:
            depth = np.linspace(init_thick, last_depth, nlayer)
            pass
        else:
            depth = np.linspace(init_thick, last_depth, nlayer-1)
            depth = np.append([0], depth)
        thicks = []
        for i in range(nlayer-1):
            thicks.append(depth[i+1]-depth[i])

        fig = plt.figure(figsize=(25,1))
        ax = fig.add_subplot(111)
        for i in range(nlayer):
            ax.plot([depth[i], depth[i]], [0,1], c='k')
    ax.set_ylim(0,1)
    ax.axes.yaxis.set_visible(False)
    ax.grid(which='both', c='#ccc')
    ax.set_xlabel('Depth (m)')

    return thicks, depth
