import numpy as np
from multiprocessing import cpu_count
from concurrent import futures
from script import ModelingToolKit as mtk
from script import EMforward as emf

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
            resp = emf.emulatte_RESOLVE(
                self.thicks, self.resistivity, self.freqs, self.nfreq, self.spans, self.height,
                vca_index=self.vca_index, add_noise=self.add_noise, noise_level=self.noise_level
                )

            #説明変数x, 目的変数yを格納
            if self.bhinx:
                xy = np.r_[resp, self.height, self.resistivity]
            else:
                xy = np.r_[resp, self.resistivity]

            xy_list.append(xy)
        
        xy_list = np.array(xy_list)
        return xy_list

class VTEM1D:
    def __init__(self):
        return None

class SkyTEM1D:
    def __init__(self):
        return None

class HeliTEM1D:
    def __init__(self):
        return None