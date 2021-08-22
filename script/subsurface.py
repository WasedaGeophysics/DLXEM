import numpy as np

class Subsurface1D:
    """
    地下構造（層厚, 比抵抗）を決定する
    """
    def __init__(self, thicks, rmean, rscat, mode):
        self.thickness = thicks
        self.res_min = 1e-1
        self.res_max = 1e5
        self.rmean = rmean
        self.rscat = rscat
        if mode == 'smooth_mix':
            self.resistivity = self.smooth_mix()
        elif mode == 'ymtmt':
            self.resistivity = np.array(self.ymtmt())
        elif mode == 'normal':
            self.resistivity = self.normal()

    def smooth_mix(self):
        size = len(self.thickness) + 1
        # Random Auto Configuration
        normal, fixed, activation = np.random.randint(0, 2, 3)
        # 移動平均による平滑化の回数
        smooth_iter = np.random.choice([2, 3, 5, 10, 20, 100])
        # 層数分の指数を乱数生成
        if normal:
            #正規分布
            exponent = np.random.randn(size) + 2
        else:
            #一様分布
            exponent = np.log10(self.res_max/self.res_min) * np.random.rand(size) + np.log10(self.res_min)

        # 初期値の固定点を第1層からn-1層までの間に１点設ける（確率で異常値付与）
        exponent0 = exponent.copy()
        fixed_index = np.random.randint(1, size-1)
        for i in range(smooth_iter):
            exponent = self.movearg(exponent)
            if fixed:
                exponent[fixed_index] = exponent0[fixed_index]

        # 活性化
        if activation:
            exponent = exponent + 0.6 * np.tanh((exponent - 2) / 3)
        res = 10 ** exponent
        return res

   
    def ymtmt(self):
        layer_num = len(self.thickness)
        # 実質、何層構造か決める
        thickness_num = np.random.randint(1, 4)
        if thickness_num == 1:
            res = self.random_resistivity_logscale(1) * layer_num
        elif thickness_num == 2:
            divider = np.random.randint(1, layer_num, 1)
            res = self.random_resistivity_logscale(1) * divider[0] + self.random_resistivity_logscale(1) * (layer_num - divider[0])
        elif thickness_num == 3:
            divider = self.random_int_nolap(thickness_num, layer_num)
            res = self.random_resistivity_logscale(1) * divider[0]\
                + self.random_resistivity_logscale(1) * (divider[1] - divider[0])\
                + self.random_resistivity_logscale(1) * (layer_num - divider[1])
        else:
            print('layer num error!')
            res = None
        return res

    def normal(self):
        size = len(self.thickness) + 1
        exponent = np.array([])
        # 移動平均による平滑化の回数
        smooth_iter = np.random.choice([0, 0, 1, 1, 5])
        abnormal_std = [0.7, 1.0, 1.3]
        natural_std = [0.1, 0.3, 0.5]
        level = self.rmean + 2.0 * self.rscat * (np.random.rand() - 0.5)
        while True:
            count = len(exponent)
            empty = size - count
            fill = int(np.random.lognormal(1.2, 0.65)) + 1
            if fill <= empty:
                abnormal = np.random.choice([False, True], p=[0.7, 0.3])
                if abnormal:
                    normal_std = np.random.choice(abnormal_std)
                else:
                    normal_std = np.random.choice(natural_std)
                exp_add = np.ones(fill) * (np.random.normal(level, normal_std))
                exponent = np.append(exponent, exp_add)
                empty -= fill
            else:
                continue
            if empty == 0:
                break
        for i in range(smooth_iter):
            exponent = self.movearg(exponent)
        res = 10 ** exponent
        return res

    def random_resistivity_logscale(self, num):
        """
        対数間隔でランダムに乱数を生成する
        :return:
        """
        res_index = np.log10(self.res_max / self.res_min) * np.random.rand(num) + np.log10(self.res_min)
        res = 10 ** res_index
        return list(res)

    @staticmethod
    def random_int_nolap(divider_num, layer_num):
        """
        ダブりなしのint型乱数を生成する
        :param num:
        :return:
        """
        random_list = []
        list_num = 0
        while list_num < divider_num:
            random = np.random.randint(1, layer_num+1)
            if random not in random_list:
                random_list.append(random)
            list_num = len(random_list)
        random_list.sort()
        return random_list
    
    @staticmethod
    def movearg(x):
        span = 3
        length = len(x)
        y = x.copy()
        edge = span // 2
        # 端は近傍３層の加重平均
        y[0] = (y[0]*3 + y[1]*2 + y[2]) / 6
        y[-1] = (y[-1]*3 + y[-2]*2 + y[-3]) /6
        if span - length >= 1:
            pass
        elif span % 2 == 0:
            for i in range(edge, length-edge):
                y[i] = (sum(x[i-edge:i+edge])/span + sum(x[i-edge+1:i+edge+1])/span)/2
        else:
            for i in range(edge, length-edge):
                y[i] = sum(x[i-edge:i+edge+1])/span
        return y
