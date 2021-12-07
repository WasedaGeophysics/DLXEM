import numpy as np
import matplotlib.pyplot as plt
from script import emforward as emf


class ResolvePlot:
    @staticmethod
    def resistivity_step(ax, thicks, pred_res, true_res, height, log_depth=False):
        thicks_add = [*thicks, thicks[-1]]
        thicks_add2 = [*thicks_add, 1]
        depth = [1e-5, *np.cumsum(thicks_add)]
        pr = np.array([*pred_res, pred_res[-1]])
        tr = np.array([*true_res, true_res[-1]])
        abs_err = pr-tr
        abs_err_p = []
        abs_err_n = []
        ii = 0
        for err in abs_err:
            if err < 0:
                abs_err_p.append(0)
                abs_err_n.append(-err)
            else:
                abs_err_p.append(err)
                abs_err_n.append(0)

        ax.barh(depth, abs_err_p, thicks_add2, align='edge', color='#8df', edgecolor='#8df', alpha=0.3, label='difference')
        ax.barh(depth, abs_err_n, thicks_add2, align='edge', color='#8df', edgecolor='#8df', alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlim(1e-2, 1e6)
        ax.set_xlabel('difference of resistivity ${\mathrm{(\Omega \cdot m)}}$')
        ax.legend(loc='upper right')

        ax.step(tr, depth, c='k', linewidth=0.6, label='label')
        ax.step(pr, depth, c='r', linewidth=0.8, label='predict')
        ax.set_xscale('log')
        ax.set_xlim(1e-2, 1e6)
        ax.set_xlabel('resistivity ${\mathrm{(\Omega \cdot m)}}$')
        ax.set_ylim(0, depth[-1])
        ax.set_ylabel('depth[m]')
        if log_depth:
            ax.set_yscale('log')
            ax.set_ylim(depth[1]/1.25, depth[-1])
        ax.invert_yaxis()
        ax.grid(which='major',color='#ccc',linestyle='-')
        ax.grid(which='minor',color='#eee',linestyle='--')
        ax.legend(loc='lower right')
        ax.set_title('Resistivity Structure (height : {} m)'.format(round(height, 2)))

    

    #== POSTER ===================#
    @classmethod
    def models_poster25(cls, thicks, pred_res, true_res, height, numbers, log_depth=False):
        fig, ax = plt.subplots(5, 5, figsize=(25,25))
        for i in range(5):
            for j in range(5):
                k = 5 * i + j
                rmspe = np.sqrt(np.sum(((pred_res[k]-true_res[k])/true_res[k])**2)/len(true_res[j]))
                cls.resistivity_step(ax[i,j], thicks, pred_res[k], true_res[k], log_depth=log_depth)
                ax[i,j].set_title('No.{}  TC = {} m , RMSPE = {} '.format(numbers[k], np.round(height[k], 1), np.round(rmspe, 3)))
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        return fig

