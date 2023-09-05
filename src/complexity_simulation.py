import time

from ClassFractionalBrownianNoise import FractionalBrownianNoise

from utils.math_functions import np
from utils.plotting_functions import plt


def plotting_complexity():
    N_samples = 1000
    av_factor = 10
    offset = 10
    H = 0.7
    fbn = FractionalBrownianNoise(H)
    times = [0] * N_samples
    for i in range(offset, N_samples):
        for _ in range(av_factor):
            startT = time.perf_counter()
            fbn.hosking_simulation(i)
            endT = time.perf_counter()
            times[i] += endT - startT  # Heuristic for computational complexity
    times = np.array(times)
    times /= float(av_factor)
    plt.plot(np.arange(10, N_samples), times[offset:])
    plt.show()
