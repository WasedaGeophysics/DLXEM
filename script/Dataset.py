import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent import futures
from script import Modeler as md
from script import w1dem as w1

class Resolve1D:
    def __init__(
            self, 
            size, thicks, brlim, bhlim, freq, sep, vca_index=3,
            add_noise=False, noise_level=None, model_generator='default'):
        self.size               = size
        # Geophysical subsurface model
        self.thicks             = thicks
        self.brlim             = brlim
        self.bhlim              = bhlim
        self.model_generator    = model_generator
        # Unique in RESOLVE system
        self.freq = freq
        self.sep = sep
        self.vca_index = vca_index
        # Preference
        self.add_noise          = add_noise
        self.noise_level        = noise_level

    def proceed(self, func=None):
        # タスクはコア数の倍に設定
        nsplit = 8
        # CPUのコア数を最大プロセス数とする
        ncpu = cpu_count()
        iters = np.split(np.arange(self.size), nsplit)
        func = self.task
        with futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
            result_maker = executor.map(func, iters)
        result = np.vstack([ans for ans in result_maker])
        return result

    def task(self, iters):
        # 説明変数Xと目的変数Yのデータセット
        xy_list = []

        for i in iters:
            # 層厚固定で比抵抗構造をランダム生成
            self.resistivity = md.resistivity1D(self.thicks, self.brlim, self.model_generator)

            #曳航高度をランダム生成
            self.height = (self.bhlim[1]-self.bhlim[0]) * np.random.rand() + self.bhlim[0]

            #RESOLVEのノイズ付応答を計算
            resp = self.forward1D()

            #説明変数x, 目的変数yを格納
            xy = np.r_[resp, self.height, self.resistivity]
            xy_list.append(xy)
        
        xy_list = np.array(xy_list)
        return xy_list

    def forward1D(self):
        """
        return : ndarray 
            [
                Re(hz1), Re(hz2), Re(hz3), Re(hz4), Re(hz5), Re(hz6),
                Im(hz1), Im(hz2), Im(hz3), Im(hz4), Im(hz5), Im(hz6)
            ]
        """
        #フォワード計算
        resistivity = self.resistivity
        thickness = self.thicks
        ry = [0]
        rz = [-self.height]
        tx = [0]
        ty = [0]
        tz = [-self.height]
        hankel_filter = 'werthmuller201'
        fdtd = 1
        dbdt = 1
        dipole_mom = 1
        displacement_current = False

        em = []

        # HCP応答の計算
        for i in range(6):
            f = np.array([self.freq[i]])
            rx = [self.sep[i]]
            plot_number = len(f)

            if i == self.vca_index:
                w1fdem = w1.Fdem(rx, ry, rz, tx, ty, tz, resistivity, thickness, hankel_filter, fdtd, dbdt, plot_number
                                , f, displacement_current=displacement_current)
                resp = w1fdem.hmdx(dipole_mom)[0]['h_x']
            else:
                w1fdem = w1.Fdem(rx, ry, rz, tx, ty, tz, resistivity, thickness, hankel_filter, fdtd, dbdt, plot_number
                                , f, displacement_current=displacement_current)
                resp = w1fdem.vmd(dipole_mom)[0]['h_z']

            em.append(resp[0])

        resp = np.array(em)
        rx = np.array(self.sep)

        #１次磁場、2次磁場をppmに変換
        primary_field = - 1 / (4 * np.pi * rx ** 3)
        real_total_field = np.real(resp)
        imag_secondary_field = np.imag(resp)
        real_secondary_field = real_total_field - primary_field
        real_ppm = real_secondary_field / primary_field * 1e6
        imag_ppm = imag_secondary_field / primary_field * 1e6

        # bookpurnongのそれぞれの周波数のノイズレベル Christensen(2009)
        # bookpurnong_noise_levels = [10, 10, 20, 40, 50]

        # ノイズ付加
        if self.add_noise:
            for index, noise_level in enumerate(self.noise_level):
                noise = np.random.normal(0, noise_level)
                real_ppm[index] = real_ppm[index] + noise
                imag_ppm[index] = imag_ppm[index] + noise

        resp = np.hstack([real_ppm, imag_ppm])
        return resp
