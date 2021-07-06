import numpy as np
import matplotlib.pyplot as plt
from dlxem import forward

class resolve:
    #== SUMPLOT =========================#
    @classmethod
    def sumplot(cls, thicks, pred_res, true_res, height, span, freqs, cfreq_range, orig_emf, ppm=True, noised=False, log_depth=False):
        fig = plt.figure(figsize=(25, 7), dpi=100)
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        cls.resistivity_step(ax1, thicks, pred_res, true_res, log_depth=log_depth)

        cfreq_min = cfreq_range[0]
        cfreq_max = cfreq_range[1]
        cfreq_size = 300
        cfreqs = np.logspace(cfreq_min, cfreq_max, cfreq_size)
        freq_size = len(freqs)

        true_emf = forward.resolve(thicks, true_res, height, span, freqs, add_noise=noised, to_ppm=ppm)
        pred_emf = forward.resolve(thicks, pred_res, height, span, freqs, add_noise=noised, to_ppm=ppm)
        true_cemf = forward.resolve(thicks, true_res, height, span, cfreqs, add_noise=noised, to_ppm=ppm)
        pred_cemf = forward.resolve(thicks, pred_res, height, span, cfreqs, add_noise=noised, to_ppm=ppm)

        if ppm == False:
            primary_field = -1 / (4 * np.pi * span ** 3)
            real_ppm = orig_emf[:freq_size]
            imag_ppm = orig_emf[freq_size:]
            raw_real = real_ppm * primary_field * 1e-6 + primary_field
            raw_imag = imag_ppm * primary_field * 1e-6
            orig_emf = np.hstack([raw_real, raw_imag])

        cls.emfield(ax2, ax3, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=noised)
        return fig

    @staticmethod
    def resistivity_step(ax, thicks, pred_res, true_res, log_depth=False):
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

        ax.barh(depth, abs_err_p, thicks_add2, align='edge', color='#8ed', edgecolor='#8ed', alpha=0.3, label='difference')
        ax.barh(depth, abs_err_n, thicks_add2, align='edge', color='#8ed', edgecolor='#8ed', alpha=0.3)
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
        ax.set_title('${ρ_{\mathrm{pred}}}$ and ${ρ_{\mathrm{true}}}$')

    @staticmethod
    def emfield(ax1, ax2, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=False):
        cfreqsize = len(cfreqs)
        freqsize = len(freqs)
        thicks = []
        for i in range(cfreqsize-1):
            thicks.append(cfreqs[i+1] - cfreqs[i])
        thicks = np.array([*thicks, thicks[-1]])
        emf_err = abs(pred_emf - true_emf) / abs(true_emf) * 100
        cemf_err = (pred_cemf - true_cemf) / abs(true_cemf) * 100
        cemf_err_p = []
        cemf_err_n = []

        for err in cemf_err:
            if err < 0:
                cemf_err_p.append(0)
                cemf_err_n.append(-err)
            else:
                cemf_err_p.append(err)
                cemf_err_n.append(0)

        if noised:
            ax1.plot(freqs, orig_emf[:freqsize], 'C2', marker='x', linewidth=0, label='true / original noised')
            ax1.plot(freqs, orig_emf[:freqsize], 'C2', marker='x', linewidth=0)
            ax2.plot(freqs, orig_emf[freqsize:], 'C2', marker='x', linewidth=0, label='true / original noised')
            ax2.plot(freqs, -orig_emf[freqsize:], 'C2', marker='x', linewidth=0)

        ax1.plot(cfreqs, true_cemf[:cfreqsize], 'k', linewidth=0.75, label='true / denoised')
        ax1.plot(cfreqs, -true_cemf[:cfreqsize], 'k', linewidth=0.75)
        ax1.plot(freqs, true_emf[:freqsize], 'k', marker='.', linewidth=0)
        ax1.plot(freqs, -true_emf[:freqsize], 'k', marker='.', linewidth=0)
        ax2.plot(cfreqs, true_cemf[cfreqsize:], 'k', linewidth=0.75, label='true / denoised')
        ax2.plot(cfreqs, -true_cemf[cfreqsize:], 'k', linewidth=0.75)
        ax2.plot(freqs, true_emf[freqsize:], 'k', marker='.', linewidth=0)
        ax2.plot(freqs, -true_emf[freqsize:], 'k', marker='.', linewidth=0)

        ax1.plot(cfreqs, pred_cemf[:cfreqsize], 'r', linewidth=0.75, label='predicted / denoised')
        ax1.plot(cfreqs, -pred_cemf[:cfreqsize], 'r', linewidth=0.75)
        ax1.plot(freqs, pred_emf[:freqsize], 'r', marker='+', linewidth=0)
        ax1.plot(freqs, -pred_emf[:freqsize], 'r', marker='+', linewidth=0)
        ax2.plot(cfreqs, pred_cemf[cfreqsize:], 'r', linewidth=0.75, label='predicted / denoised')
        ax2.plot(cfreqs, -pred_cemf[cfreqsize:], 'r', linewidth=0.75)
        ax2.plot(freqs, pred_emf[freqsize:], 'r', marker='+', linewidth=0)
        ax2.plot(freqs, -pred_emf[freqsize:], 'r', marker='+', linewidth=0)

        ax1b = ax1.twinx()
        ax2b = ax2.twinx()

        ax1b.bar(cfreqs, cemf_err_p[:cfreqsize], thicks, align='edge', color='#8ed', alpha=0.3, label='Relative Error')
        ax1b.bar(cfreqs, cemf_err_n[:cfreqsize], thicks, align='edge', color='#8ed', alpha=0.3)
        ax1b.plot(freqs, emf_err[:freqsize], c='#8ed', marker='.', linewidth=0)
        ax2b.bar(cfreqs, cemf_err_p[cfreqsize:], thicks, align='edge', color='#8ed', alpha=0.3, label='Relative Error')
        ax2b.bar(cfreqs, cemf_err_n[cfreqsize:], thicks, align='edge', color='#8ed', alpha=0.3)
        ax2b.plot(freqs, emf_err[freqsize:], c='#8ed', marker='.', linewidth=0)

        for ax in [ax1b, ax2b]:
            ax.set_ylim(0, 300)
            ax.set_ylabel('Relative Error (%)')

        ax1.set_title('Real Part')
        for ax in [ax1, ax2]:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.set_xlim(cfreqs[0], cfreqs[-1])
            ax.set_ylim(1e-4, 1e8)
            ax.set_xlabel('Frequecy (Hz)')
            ax.set_ylabel('Secondary field $h_z$ (ppm)')
            ax.grid(which='major',color='#ccc',linestyle='-')
            #ax.grid(which='minor',color='#eee',linestyle='')
        ax1.set_title('Real Part')
        ax2.set_title('Imaginary Part')

    @staticmethod
    def emfield_error(ax1, ax2, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=False):
        cfreqsize = len(cfreqs)
        freqsize = len(freqs)
        emf_err = abs(pred_emf - true_emf) / abs(true_emf) * 100
        cemf_err = (pred_cemf - true_cemf) / abs(true_cemf) * 100
        oemf_err = abs(pred_emf - orig_emf) / abs(orig_emf) * 100

        ax1.plot(cfreqs, cemf_err[:cfreqsize], c='navy', label='vs denoized')
        ax1.plot(freqs, emf_err[:freqsize], c='navy', marker='.', linewidth=0)
        ax1.plot(cfreqs, -cemf_err[:cfreqsize], c='navy', linestyle='--')
        
        ax2.plot(cfreqs, cemf_err[cfreqsize:], c='navy', label='vs denoized')
        ax2.plot(freqs, emf_err[freqsize:], c='navy', marker='.', linewidth=0)
        ax2.plot(cfreqs, -cemf_err[cfreqsize:], c='navy', linestyle='--')

        if noised:
            ax1.plot(freqs, oemf_err[:freqsize], 'C2', marker='x', linewidth=0, label='vs original noised')
            ax2.plot(freqs, oemf_err[freqsize:], 'C2', marker='x', linewidth=0, label='vs original noised')

        for ax in [ax1, ax2]:
            ax.set_xscale('log')
            ax.set_ylim(-0.1,100)
            ax.set_xlabel('Frequecy (Hz)')
            ax.set_ylabel('Relative error (%)')
            ax.grid(which='major',color='#ccc',linestyle='-')
            ax.grid(which='minor',color='#eee',linestyle='--')
            ax.legend()
        
        ax1.set_title('')
        ax2.set_title('')

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
