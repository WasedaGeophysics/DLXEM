import numpy as np

def resistivity1D(thicks, brlim, model_generator):
    """
    thicks : list, array-like
        list of thickness in each layer
    brlim : list [min, max]
        limits of resistivity range (Ohm-m)
    """
    if model_generator == "default":
        size = len(thicks) + 1
        lower = np.log10(brlim[0])
        upper = np.log10(brlim[1])
        baseres = [np.random.rand() * (upper - lower) + lower] * size
        baseres = np.array(baseres)
        altres = np.array([])

        m = int(size//12) + 1
        k = np.exp(3/5*np.exp(-1/size))

        sigma = np.sqrt(2/3*np.log(k))
        mu = np.log(m*k**(2/3))
        n = int(size//50) + 1

        smooth_iter = np.random.choice([1, 1, 1, 2, 5]) * n
        abnormal_std = [0.5, 0.8]
        natural_std = [0.1, 0.3]
        while True:
            count = len(altres)
            empty = size - count
            if empty <= size*.05:
                fill = empty
            else:
                fill = int(np.random.lognormal(mu, sigma))
            if fill == 0:
                continue
            if fill <= empty:
                abnormal = np.random.choice([False, True], p=[0.7, 0.3])
                if abnormal:
                    normal_std = np.random.choice(abnormal_std)
                else:
                    normal_std = np.random.choice(natural_std)
                exp_add = np.ones(fill) * (np.random.normal(0, normal_std))
                altres = np.append(altres, exp_add)
                empty -= fill
            else:
                continue
            if empty == 0:
                break
        exponent = baseres + altres
        for i in range(smooth_iter):
            exponent = movearg(exponent)
        res = 10 ** exponent
        return res
    
def movearg(x):
    span = 3
    length = len(x)
    y = x.copy()
    edge = span // 2

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