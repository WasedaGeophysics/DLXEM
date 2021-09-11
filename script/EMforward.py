import numpy as np
from script.emulatte import forward as fwd

def emulatte_RESOLVE(
        thicks, resistivity, freqs, nfreq, spans, height, 
        vca_index=3, add_noise=False, noise_level=None
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
        dipole_moment = 1
        displacement_current = False

        model = fwd.model(thicks)
        model.add_resistivity(resistivity)
    
        fields = []
        primary_fields = []

        
        # HCP, VCA応答の計算
        for i in range(nfreq):
            f = np.array([freqs[i]])
            rc = [-spans[i], 0, -height]
            if (nfreq == 6) and (i ==  vca_index):
                hmdx = fwd.transmitter("HMDx", f, dipole_moment=dipole_moment)
                model.locate(hmdx, tc, rc)
                resp, _ = model.emulate(hankel_filter=hankel_filter)
                resp = resp['h_x'][0]
                primary_field = dipole_moment / (2 * np.pi * spans[i] ** 3)
            else:
                vmd = fwd.transmitter("VMD", f, dipole_moment=dipole_moment)
                model.locate(vmd, tc, rc)
                resp, _ = model.emulate(hankel_filter=hankel_filter)
                resp = resp['h_z'][0]
                primary_field = - dipole_moment / (4 * np.pi * spans[i] ** 3)
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
        if add_noise:
            for index, nlv in enumerate(noise_level):
                inphnoise = np.random.normal(0, nlv)
                quadnoise = np.random.normal(0, nlv)
                real_ppm[index] = real_ppm[index] + inphnoise
                imag_ppm[index] = imag_ppm[index] + quadnoise

        resp = np.hstack([real_ppm, imag_ppm])
        return resp