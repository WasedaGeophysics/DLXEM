import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent import futures
from dlxem import subsurface as sf
from dlxem import forward as fwd

class Resolve1D:
    def __init__(
            self, 
            thicks, res_range, height_range, freqs, span,
            to_ppm=True, add_noise=False, noise_level=None,
            random_mode='ymtmt', ground_char = 'r'):
        self.thicks = thicks
        self.res_range = res_range
        self.height_range = height_range
        self.freqs = freqs
        self.span = span
        self.to_ppm = to_ppm
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.random_mode = random_mode
        self.ground_char = ground_char

    def multi_process(self, size, nsplit, func=None):
        if func == None:
            func = self.default
        iters = np.split(np.arange(size), nsplit)
        ncpu = cpu_count()
        with futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
            result_maker = executor.map(func, iters)

        result = np.vstack([ans for ans in result_maker])
        return result

    def default(self, iters):
        height_max = self.height_range[1]
        height_min = self.height_range[0]

        xy_list = []

        for i in iters:
            # 層厚固定で比抵抗構造をランダム生成
            hml = sf.Subsurface1D(self.thicks, self.random_mode)

            #曳航高度をランダム生成
            height = (height_max - height_min) * np.random.rand() + height_min

            #RESOLVEのノイズ付応答を計算
            resp = fwd.resolve(hml.thickness, hml.resistivity, height, self.span, self.freqs, 
                            add_noise=self.add_noise, to_ppm=self.to_ppm, noise_level=self.noise_level)

            #説明変数x, 目的変数yを格納
            xy = np.r_[resp, height, hml.resistivity]
            xy_list.append(xy)
        
        xy_list = np.array(xy_list)
        return xy_list