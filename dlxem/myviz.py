import numpy as np
import matplotlib.pyplot as plt
from dlxem import forward

class resolve:
    #== SUMPLOT =========================#
    @classmethod
    def sumplot(cls, thicks, pred_res, true_res, height, span, freqs, cfreq_size, orig_emf, ppm=True, noised=False, log_depth=False):
        fig = plt.figure(figsize=(18, 21), dpi=100)
        ax1 = fig.add_subplot(3,2,1)
        ax2 = fig.add_subplot(3,2,2)
        ax3 = fig.add_subplot(3,2,3)
        ax4 = fig.add_subplot(3,2,4)
        ax5 = fig.add_subplot(3,2,5)
        ax6 = fig.add_subplot(3,2,6)

        cls.resistivity_step(ax1, thicks, pred_res, true_res, log_depth=log_depth)
        cls.resistivity_error_step(ax2, thicks, pred_res, true_res, log_depth=log_depth)

        cfreq_min = 1
        cfreq_max = 9
        cfreqs = np.logspace(cfreq_min, cfreq_max, cfreq_size)
        freq_size = len(freqs)

        if ppm == False:
            true_emf = forward.resolve(thicks, true_res, height, span, freqs, add_noise=noised, to_ppm=ppm)
            pred_emf = forward.resolve(thicks, pred_res, height, span, freqs, add_noise=noised, to_ppm=ppm)
            true_cemf = forward.resolve(thicks, true_res, height, span, cfreqs, add_noise=noised, to_ppm=ppm)
            pred_cemf = forward.resolve(thicks, pred_res, height, span, cfreqs, add_noise=noised, to_ppm=ppm)

            primary_field = -1 / (4 * np.pi * span ** 3)
            real_ppm = orig_emf[:freq_size]
            imag_ppm = orig_emf[freq_size:]
            raw_real = real_ppm * primary_field * 1e-6 + primary_field
            raw_imag = imag_ppm * primary_field * 1e-6
            orig_emf = np.hstack([raw_real, raw_imag])

        else:
            true_emf = forward.resolve(thicks, true_res, height, span, freqs, add_noise=noised, to_ppm=ppm)
            pred_emf = forward.resolve(thicks, pred_res, height, span, freqs, add_noise=noised, to_ppm=ppm)
            true_cemf = forward.resolve(thicks, true_res, height, span, cfreqs, add_noise=noised, to_ppm=ppm)
            pred_cemf = forward.resolve(thicks, pred_res, height, span, cfreqs, add_noise=noised, to_ppm=ppm)

        cls.emfield_real(ax3, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=noised)
        cls.emfield_imag(ax5, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=noised)
        cls.emfield_error(ax4, ax6, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=noised)

        return fig

    @staticmethod
    def resistivity_step(ax, thicks, pred_res, true_res, log_depth=False):
        thicks_add = [*thicks, thicks[-1]]
        #thicks_add = thicks
        pr = [*pred_res, pred_res[-1]]
        tr = [*true_res, true_res[-1]]
        depth = [1e-5, *np.cumsum(thicks_add)]
        ax.step(tr, depth, linewidth=0.6, label='label')
        ax.step(pr, depth, linewidth=0.8, label='predict')
        ax.set_xscale('log')
        ax.set_xlim(1e-2, 1e6)
        ax.set_xlabel('resistivity ${\mathrm{(\Omega \cdot m)}}$')
        ax.set_ylim(0, depth[-1])
        ax.set_ylabel('depth[m]')
        if log_depth:
            ax.set_yscale('log')
            ax.set_ylim(10**(-1.5), depth[-1])
        ax.invert_yaxis()
        ax.grid(which='major',color='#ccc',linestyle='-')
        ax.grid(which='minor',color='#eee',linestyle='--')
        ax.legend()
        ax.set_title('${ρ_{\mathrm{pred}}}$ and ${ρ_{\mathrm{true}}}$')

    @staticmethod
    def resistivity_error_step(ax, thicks, pred_res, true_res, log_depth=False):
        thicks_add = [*thicks, thicks[-1]]
        #thicks_add = thicks
        pr = np.array([*pred_res, pred_res[-1]])
        tr = np.array([*true_res, true_res[-1]])
        depth = [1e-5, *np.cumsum(thicks_add)]
        abs_err = abs(pr-tr)

        ax.step(abs_err, depth, c='navy', linewidth=0.5, label='Absolute Error')
        #low = 10 ** (np.log10(min(abs_err))-0.1)
        #high = max(abs_err) * 1.1
        #ax.set_xlim(low, high)
        ax.set_xlim(-2, 6)
        ax.set_xscale('log')
        ax.set_xlabel('Absolute Error ${\mathrm{(\Omega \cdot m)}}$')
        ax.set_ylim(0, depth[-1])
        ax.set_ylabel('depth[m]')
        if log_depth:
            ax.set_yscale('log')
            ax.set_ylim(10**(-1.5), depth[-1])
        ax.invert_yaxis()
        ax.grid(which='major',color='#ccc',linestyle='-')
        ax.grid(which='minor',color='#eee',linestyle='--')
        ax.set_title('Absolute Error between ${ρ_{\mathrm{pred}}}$ and ${ρ_{\mathrm{true}}}$')

    @staticmethod
    def emfield_real(ax, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=False):
        cfreqsize = len(cfreqs)
        freqsize = len(freqs)

        if noised:
            ax.plot(freqs, orig_emf[:freqsize], 'C2', marker='x', linewidth=0, label='true / original noised')
            ax.plot(freqs, -orig_emf[:freqsize], 'C2', marker='x', linewidth=0)

        ax.plot(cfreqs, true_cemf[:cfreqsize], 'C0', linewidth=0.75, label='true / denoised')
        ax.plot(cfreqs, -true_cemf[:cfreqsize], 'C0--', linewidth=0.75)
        ax.plot(freqs, true_emf[:freqsize], 'C0', marker='.', linewidth=0)
        ax.plot(freqs, -true_emf[:freqsize], 'C0', marker='.', linewidth=0)

        ax.plot(cfreqs, pred_cemf[:cfreqsize], 'C1', linewidth=0.75, label='predicted / denoised')
        ax.plot(cfreqs, -pred_cemf[:cfreqsize], 'C1--', linewidth=0.75)
        ax.plot(freqs, pred_emf[:freqsize], 'C1', marker='+', linewidth=0)
        ax.plot(freqs, -pred_emf[:freqsize], 'C1', marker='+', linewidth=0)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Frequecy (Hz)')
        ax.set_ylabel('Secondary field $h_z$ (ppm)')
        ax.grid(which='major',color='#ccc',linestyle='-')
        ax.grid(which='minor',color='#eee',linestyle='--')
        ax.set_title('Real Part')

    @staticmethod
    def emfield_imag(ax, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf, noised=False):
        cfreqsize = len(cfreqs)
        freqsize = len(freqs)
        if noised:
            ax.plot(freqs, orig_emf[freqsize:], 'C2', marker='x', linewidth=0, label='true / original noised')
            ax.plot(freqs, -orig_emf[freqsize:], 'C2', marker='x', linewidth=0)

        ax.plot(cfreqs, true_cemf[cfreqsize:], 'C0', linewidth=0.75, label='true / denoised')
        ax.plot(cfreqs, -true_cemf[cfreqsize:], 'C0--', linewidth=0.75)
        ax.plot(freqs, true_emf[freqsize:], 'C0', marker='.', linewidth=0)
        ax.plot(freqs, -true_emf[freqsize:], 'C0', marker='.', linewidth=0)

        ax.plot(cfreqs, pred_cemf[cfreqsize:], 'C1', linewidth=0.75, label='predicted / denoised')
        ax.plot(cfreqs, -pred_cemf[cfreqsize:], 'C1--', linewidth=0.75)
        ax.plot(freqs, pred_emf[freqsize:], 'C1', marker='+', linewidth=0)
        ax.plot(freqs, -pred_emf[freqsize:], 'C1', marker='+', linewidth=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Frequecy (Hz)')
        ax.set_ylabel('Secondary field $h_z$ (ppm)')
        ax.grid(which='major',color='#ccc',linestyle='-')
        ax.grid(which='minor',color='#eee',linestyle='--')
        ax.set_title('Imaginary Part')

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
