import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent import futures
from script import ModelingToolKit as mtk
from script import w1dem as w1
from script.emulatte import forward as fwd

class Resolve1D:
    def __init__(
            self, 
            size, thicks, bgrlim, bhlim, freqs, spans, vca_index=3,
            add_noise=False, noise_level=None, generate_mode='default',
            bhinx = True):
        self.size               = size
        # Geophysical subsurface model
        self.thicks             = thicks
        self.bgrlim             = bgrlim
        self.bhlim              = bhlim
        self.generate_mode    = generate_mode
        # Unique in RESOLVE system
        self.freqs = freqs
        self.nfreq = len(freqs)
        self.spans = spans
        self.vca_index = vca_index
        # Preference
        self.add_noise          = add_noise
        self.noise_level        = noise_level
        self.bhinx = bhinx
        

    def proceed(self):
        # タスクはコア数の倍に設定
        ncpu = 20
        # CPUのコア数を最大プロセス数とする
        nsplit = cpu_count()
        iters = np.split(np.arange(self.size), nsplit)
        func = self.task
        with futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
            result_maker = executor.map(func, iters)
        result = np.vstack([ans for ans in result_maker])
        return result

    def task(self, iters):
        # 説明変数Xと目的変数YのDataset
        xy_list = []

        for i in iters:
            # 層厚固定で比抵抗構造をランダム生成
            self.resistivity = mtk.resistivity1D(self.thicks, self.bgrlim, self.generate_mode)

            #曳航高度をランダム生成
            self.height = (self.bhlim[1]-self.bhlim[0]) * np.random.rand() + self.bhlim[0]

            #RESOLVEのノイズ付応答を計算
            resp = self.emulatte_RESOLVE()

            #説明変数x, 目的変数yを格納
            if self.bhinx:
                xy = np.r_[resp, self.height, self.resistivity]
            else:
                xy = np.r_[resp, self.resistivity]
            xy_list.append(xy)
        
        xy_list = np.array(xy_list)
        return xy_list

    def emulatte_RESOLVE(self):
        """
        return : ndarray 
            [
                Re(HCP1), Re(HCP2), Re(HCP3), (Re(VCX)), Re(HCP4), Re(HCP5),
                Im(HCP1), Im(HCP2), Im(HCP3), (Im(VCX)), Im(HCP4), Im(HCP5),
            ]
        """
        #フォワード計算
        tc = [0, 0, -self.height]
        hankel_filter = 'werthmuller201'
        dipole_moment = 1
        displacement_current = False

        model = fwd.model(self.thicks)
        model.add_resistivity(self.resistivity)
    
        fields = []
        primary_fields = []

        
        # HCP, VCA応答の計算
        for i in range(self.nfreq):
            f = np.array([self.freqs[i]])
            rc = [-self.spans[i], 0, -self.height]
            if (self.nfreq == 6) and (i ==  self.vca_index):
                hmdx = fwd.transmitter("HMDx", f, dipole_moment=dipole_moment)
                model.locate(hmdx, tc, rc)
                resp, _ = model.emulate(hankel_filter=hankel_filter)
                resp = resp['h_x'][0]
                primary_field = dipole_moment / (2 * np.pi * self.spans[i] ** 3)
            else:
                vmd = fwd.transmitter("VMD", f, dipole_moment=dipole_moment)
                model.locate(vmd, tc, rc)
                resp, _ = model.emulate(hankel_filter=hankel_filter)
                resp = resp['h_z'][0]
                primary_field = - dipole_moment / (4 * np.pi * self.spans[i] ** 3)
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
        if self.add_noise:
            for index, noise_level in enumerate(self.noise_level):
                inphnoise = np.random.normal(0, noise_level)
                quadnoise = np.random.normal(0, noise_level)
                real_ppm[index] = real_ppm[index] + inphnoise
                imag_ppm[index] = imag_ppm[index] + quadnoise

        resp = np.hstack([real_ppm, imag_ppm])
        return resp