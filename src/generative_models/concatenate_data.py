import numpy as np

from utils import config
from utils.math_functions import generate_fBn

if __name__ == "__main__":
    h = 0.7
    td = 256
    numSamples = 200000
    data = np.load(config.ROOT_DIR + "data/six_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h, td))
    data2 = generate_fBn(H=h, T=td, S=numSamples, rng=np.random.default_rng())
    data = np.concatenate([data, data2], axis=0)
    assert(data.shape[0] == int(numSamples*4) and data.shape[1] == td)
    np.save(config.ROOT_DIR + "data/eight_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h, td), data)
