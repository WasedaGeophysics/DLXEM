import numpy as np
from dlxem import w1dem as w1

def resolve(em, sf, height):
    """
    return : ndarray 
        [Re(hz1), Re(hz2), Re(hz3), Re(hz4), Re(hz5), 
         Im(hz1), Im(hz2), Im(hz3), Im(hz4), Im(hz5), ]
    """
    #フォワード計算
    resistivity = sf.resistivity
    thickness = sf.thickness
    rx = [em.span]
    ry = [0]
    rz = [-height]
    tx = [0]
    ty = [0]
    tz = [-height]
    hankel_filter = 'werthmuller201'
    fdtd = 1
    dbdt = 1
    dipole_mom = 1
    freqs = np.array(em.freqs)
    plot_number = len(freqs)
    displacement_current = False
    w1fdem = w1.Fdem(rx, ry, rz, tx, ty, tz, resistivity, thickness, hankel_filter, fdtd, dbdt, plot_number
                        , freqs, displacement_current=displacement_current)
    resp = w1fdem.vmd(dipole_mom)[0]['h_z']

    #１次磁場、2次磁場をppmに変換
    primary_field = -1 / (4 * np.pi * rx[0] ** 3)
    real_total_field = np.real(resp)
    imag_secondary_field = np.imag(resp)
    real_secondary_field = real_total_field - primary_field
    real_ppm = real_secondary_field / primary_field * 1e6
    imag_ppm = imag_secondary_field / primary_field * 1e6

    # bookpurnongのそれぞれの周波数のノイズレベル Christensen(2009)
    # bookpurnong_noise_levels = [10, 10, 20, 40, 50]

    # ノイズ付加
    if em.add_noise:
        for index, noise_level in enumerate(em.noise_level):
            noise = np.random.normal(0, noise_level)
            real_ppm[index] = real_ppm[index] + noise
            imag_ppm[index] = imag_ppm[index] + noise

    # 1次磁場, 2次磁場に戻す
    if not em.to_ppm:
        raw_real = real_ppm * primary_field * 1e-6 + primary_field
        raw_imag = imag_ppm * primary_field * 1e-6
        resp = np.hstack([raw_real, raw_imag])
    else:
        resp = np.hstack([real_ppm, imag_ppm])
    return resp
