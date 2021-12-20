import numpy as np
from multiprocessing import cpu_count
from concurrent import futures
from . import ModelingToolKit as mtk
from . import emforward as emf

class Resolve1D:
    def __init__(
            self, 
            size, thicks, bgrlim, bhlim, freqs, spans, vca_index=3,
            add_noise=False, noise_ave=None, noise_std=None, generate_mode='default',
            ):
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
        self.add_noise = add_noise
        self.noise_ave = noise_ave
        self.noise_std = noise_std
        

    def proceed(self):
        # タスクはコア数の倍に設定?
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
            resistivity = mtk.resistivity1D(self.thicks, self.bgrlim, self.generate_mode)

            #曳航高度をランダム生成
            height = (self.bhlim[1]-self.bhlim[0]) * np.random.rand() + self.bhlim[0]

            #RESOLVEのノイズ付応答を計算
            resp = emf.emulatte_RESOLVE(
                self.thicks, resistivity, self.freqs, self.nfreq, self.spans, height,
                vca_index=self.vca_index, add_noise=self.add_noise, noise_ave=self.noise_ave, noise_std=self.noise_std
                )

            #説明変数x, 目的変数yを格納
            xy = np.r_[resp, height, resistivity]
            xy_list.append(xy)
        
        xy_list = np.array(xy_list)
        return xy_list