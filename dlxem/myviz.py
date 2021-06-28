import numpy as np
import matplotlib.pyplot as plt
from dlxem import forward

class resolve:
    def __init__(self):
        pass

    @classmethod
    def sample_summary_plot(cls, thicks, pred_res, true_res, height, span, freqs, cfreq_size, orig_emf, ppm=True):
        fig = plt.figure(figsize=(14, 21), dpi=100)
        ax1 = fig.add_subplot(3,2,1)
        ax2 = fig.add_subplot(3,2,2)
        ax3 = fig.add_subplot(3,2,3)
        ax4 = fig.add_subplot(3,2,4)
        ax5 = fig.add_subplot(3,2,5)
        ax6 = fig.add_subplot(3,2,6)

        cls.resistivity_step(ax1, thicks, pred_res, true_res, log_depth=False)
        cls.resistivity_error_step(ax2, thicks, pred_res, true_res, log_depth=False)

        cfreq_min = np.log10(min(freqs)*0.1)
        cfreq_max = np.log10(max(freqs)*10)
        cfreqs = np.logspace(cfreq_min, cfreq_max, cfreq_size)
        freq_size = len(freqs)
        if not ppm:
            true_emf = forward.resolve(thicks, true_res, height, span, freqs, add_noise=False, to_ppm=False)
            pred_emf = forward.resolve(thicks, pred_res, height, span, freqs, add_noise=False, to_ppm=False)
            true_cemf = forward.resolve(thicks, true_res, height, span, cfreqs, add_noise=False, to_ppm=False)
            pred_cemf = forward.resolve(thicks, pred_res, height, span, cfreqs, add_noise=False, to_ppm=False)

            primary_field = -1 / (4 * np.pi * span ** 3)
            real_ppm = orig_emf[:freq_size]
            imag_ppm = orig_emf[freq_size:]
            raw_real = real_ppm * primary_field * 1e-6 + primary_field
            raw_imag = imag_ppm * primary_field * 1e-6
            orig_emf = np.hstack([raw_real, raw_imag])

        else:
            true_emf = forward.resolve(thicks, true_res, height, span, freqs, add_noise=False)
            pred_emf = forward.resolve(thicks, pred_res, height, span, freqs, add_noise=False)
            true_cemf = forward.resolve(thicks, true_res, height, span, cfreqs, add_noise=False)
            pred_cemf = forward.resolve(thicks, pred_res, height, span, cfreqs, add_noise=False)

        cls.emfield_real(ax3, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf)
        cls.emfield_imag(ax5, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf)
        cls.emfield_error(ax4, ax6, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf)

        return fig

    def resistivity_step(ax, thicks, pred_res, true_res, log_depth=False):
        thicks_add = [*thicks, thicks[-1]]
        #thicks_add = thicks
        pr = [*pred_res, pred_res[-1]]
        tr = [*true_res, true_res[-1]]
        depth = [0, *np.cumsum(thicks_add)]
        ax.step(pr, depth, label='predict')
        ax.step(tr, depth, label='label')
        ax.set_xscale('log')
        ax.set_xlim(1e-2, 1e6)
        ax.set_xlabel('resistivity ${\mathrm{(\Omega \cdot m)}}$')
        ax.set_ylim(0, depth[-1])
        ax.set_ylabel('depth[m]')
        if log_depth:
            ax.set_yscale('log')
            ax.set_ylim(1e-3, depth[-1])
        ax.invert_yaxis()
        ax.legend()
        ax.set_title('${ρ_{\mathrm{pred}}}$ and ${ρ_{\mathrm{true}}}$')

    @staticmethod
    def resistivity_error_step(ax, thicks, pred_res, true_res, log_depth=False):
        thicks_add = [*thicks, thicks[-1]]
        #thicks_add = thicks
        pr = np.array([*pred_res, pred_res[-1]])
        tr = np.array([*true_res, true_res[-1]])
        depth = [0, *np.cumsum(thicks_add)]
        abs_err = abs(pr-tr)

        ax.step(abs_err, depth)
        low = 10 ** (int(np.log10(min(abs_err)))-2)
        high = max(abs_err) * 1.1
        ax.set_xlim(-high*0.05, high)
        #ax.set_xscale('log')
        ax.set_xlabel('Absolute Error ${\mathrm{(\Omega \cdot m)}}$')
        ax.set_ylim(0, depth[-1])
        ax.set_ylabel('depth[m]')
        if log_depth:
            ax.set_yscale('log')
            ax.set_ylim(1e-3, depth[-1])
        ax.invert_yaxis()
        ax.set_title('Absolute Error between ${ρ_{\mathrm{pred}}}$ and ${ρ_{\mathrm{true}}}$')
        
    @staticmethod
    def emfield_imag(ax, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf):
        cfreqsize = len(cfreqs)
        freqsize = len(freqs)
        ax.plot(cfreqs, pred_cemf[cfreqsize:], 'C0', linewidth=0.5, label='predicted')
        ax.plot(cfreqs, -pred_cemf[cfreqsize:], 'C0--', linewidth=0.5)
        ax.plot(freqs, pred_emf[freqsize:], 'C0', marker='+', linewidth=0)
        ax.plot(freqs, -pred_emf[freqsize:], 'C0', marker='+', linewidth=0)
        ax.plot(cfreqs, true_cemf[cfreqsize:], 'C1', linewidth=0.5, label='denoised')
        ax.plot(cfreqs, -true_cemf[cfreqsize:], 'C1--', linewidth=0.5)
        ax.plot(freqs, true_emf[freqsize:], 'C1', marker='.', linewidth=0)
        ax.plot(freqs, -true_emf[freqsize:], 'C1', marker='.', linewidth=0)
        ax.plot(freqs, orig_emf[freqsize:], 'C2', marker='x', linewidth=0, label='original noised')
        ax.plot(freqs, -orig_emf[freqsize:], 'C2', marker='x', linewidth=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Frequecy (Hz)')
        ax.set_ylabel('Secondary field $h_z$ (ppm)')
        ax.set_title('Imaginary Part')

    @staticmethod
    def emfield_real(ax, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf):
        cfreqsize = len(cfreqs)
        freqsize = len(freqs)
        ax.plot(cfreqs, pred_cemf[:cfreqsize], 'C0', linewidth=0.5, label='predicted')
        ax.plot(cfreqs, -pred_cemf[:cfreqsize], 'C0--', linewidth=0.5)
        ax.plot(freqs, pred_emf[:freqsize], 'C0', marker='+', linewidth=0)
        ax.plot(freqs, -pred_emf[:freqsize], 'C0', marker='+', linewidth=0)
        ax.plot(cfreqs, true_cemf[:cfreqsize], 'C1', linewidth=0.5, label='denoised')
        ax.plot(cfreqs, -true_cemf[:cfreqsize], 'C1--', linewidth=0.5)
        ax.plot(freqs, true_emf[:freqsize], 'C1', marker='.', linewidth=0)
        ax.plot(freqs, -true_emf[:freqsize], 'C1', marker='.', linewidth=0)
        ax.plot(freqs, orig_emf[:freqsize], 'C2', marker='x', linewidth=0, label='original noised')
        ax.plot(freqs, -orig_emf[:freqsize], 'C2', marker='x', linewidth=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Frequecy (Hz)')
        ax.set_ylabel('Secondary field $h_z$ (ppm)')
        ax.set_title('Real Part')

    @staticmethod
    def emfield_error(ax1, ax2, freqs, cfreqs, pred_emf, true_emf, pred_cemf, true_cemf, orig_emf):
        cfreqsize = len(cfreqs)
        freqsize = len(freqs)
        emf_err = abs(pred_emf - true_emf) / abs(true_emf) * 100
        cemf_err = (pred_cemf - true_cemf) / abs(true_cemf) * 100
        oemf_err = abs(pred_emf - orig_emf) / abs(orig_emf) * 100

        ax1.plot(cfreqs, cemf_err[:cfreqsize], 'C1', label='vs denoized')
        ax1.plot(freqs, emf_err[:freqsize], 'C1', marker='.', linewidth=0)
        ax1.plot(cfreqs, -cemf_err[:cfreqsize], 'C1--')
        ax1.plot(freqs, oemf_err[:freqsize], 'C2', marker='x', linewidth=0, label='vs original noised')
        
        ax2.plot(cfreqs, cemf_err[cfreqsize:], 'C1', label='vs denoized')
        ax2.plot(freqs, emf_err[freqsize:], 'C1', marker='.', linewidth=0)
        ax2.plot(cfreqs, -cemf_err[cfreqsize:], 'C1--')
        ax2.plot(freqs, oemf_err[freqsize:], 'C2', marker='x', linewidth=0, label='vs original noised')

        for ax in [ax1, ax2]:
            ax.set_xscale('log')
            ax.set_ylim(-0.1,100)
            ax.set_xlabel('Frequecy (Hz)')
            ax.set_ylabel('Relative error (%)')
            ax.legend()
        
        ax1.set_title('')
        ax2.set_title('')