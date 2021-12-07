import numpy as np
from script.emulatte import forward as fwd

def emulatte_RESOLVE(
        thicks, resistivity, freqs, nfreq, spans, height, 
        vca_index=None, add_noise=False, noise_ave=None, noise_std=None
        ):
        """
        return : ndarray 
            [
                Re(HCP1), Re(HCP2), Re(HCP3), (Re(VCX)), Re(HCP4), Re(HCP5),
                Im(HCP1), Im(HCP2), Im(HCP3), (Im(VCX)), Im(HCP4), Im(HCP5),
            ]
        """
        #フォワード計算
        tc = [0, 0, -height]
        hankel_filter = 'werthmuller201'
        moment = 1
        displacement_current = False
        res = np.append(2e14, resistivity)

        model = fwd.model(thicks)
        model.set_properties(res=res)
    
        fields = []
        primary_fields = []

        
        # HCP, VCA応答の計算
        for i in range(nfreq):
            f = np.array([freqs[i]])
            rc = [-spans[i], 0, -height]
            # VCAあり
            if (nfreq == 6) and (i ==  vca_index):
                hmdx = fwd.transmitter("HMDx", f, moment=moment)
                model.locate(hmdx, tc, rc)
                resp = model.emulate(hankel_filter=hankel_filter)
                resp = resp['h_x'][0]
                primary_field = moment / (2 * np.pi * spans[i] ** 3)
            # VCAなし
            else:
                vmd = fwd.transmitter("VMD", f, moment=moment)
                model.locate(vmd, tc, rc)
                resp = model.emulate(hankel_filter=hankel_filter)
                resp = resp['h_z'][0]
                primary_field = - moment / (4 * np.pi * spans[i] ** 3)
            fields.append(resp)
            primary_fields.append(primary_field)

        fields = np.array(fields)
        primary_fields = np.array(primary_fields)

        #１次磁場、2次磁場をppmに変換
        inph_total_field = np.real(fields)
        quad_secondary_field = np.imag(fields)
        inph_secondary_field = inph_total_field - primary_fields
        real_ppm = abs(inph_secondary_field / primary_fields) * 1e6
        imag_ppm = abs(quad_secondary_field / primary_fields) * 1e6
        # bookpurnongのそれぞれの周波数のノイズレベル Christensen(2009)

        # ノイズ付加
        add = np.random.choice([True, False], p=[0.7, 0.3])
        if (add_noise & add):
            noise = [nlv for nlv in zip(noise_ave, noise_std)]
            for index, nlv in enumerate(noise):
                inphnoise = np.random.normal(nlv[0], nlv[1])
                quadnoise = np.random.normal(nlv[0], nlv[1])
                real_ppm[index] = real_ppm[index] + inphnoise
                imag_ppm[index] = imag_ppm[index] + quadnoise

        resp = np.hstack([real_ppm, imag_ppm])
        return resp