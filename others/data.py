import numpy as np
import sklearn.datasets

#### data generators are derived from https://github.com/lukovnikov/improved_wgan_training/blob/master/gan_toy.py
# Copyright (c) 2017 Ishaan Gulrajani
# Released under the MIT license
# https://github.com/lukovnikov/improved_wgan_training/blob/master/LICENSE

def prepare_swissroll_data(BATCH_SIZE=1000, seed=None, noise=.5):
    data = sklearn.datasets.make_swiss_roll(
                    n_samples=BATCH_SIZE,
                    noise=noise,
                    random_state=seed
                )[0]
    data = data.astype('float32')[:, [0, 2]]
    #print("", np.std(data))
    data /= 6.865#np.std(data)
    return data

def prepare_25gaussian_data(BATCH_SIZE=1000, seed=None, noise=.05):
    if seed is not None:
        rng = np.random.default_rng(seed)
        generate = lambda noise: rng.normal(0, 1, size=(2,))*noise
    else:
        generate = lambda noise: np.random.normal(0, 1, size=(2,))*noise
    data = []
    for i in range(BATCH_SIZE//25):
        for x in range(-2, 3):
            for y in range(-2, 3):
                point = generate(noise=noise)
                point[0] += 2*x
                point[1] += 2*y
                data.append(point)
    data = np.array(data, dtype=np.float32)
    np.random.shuffle(data)
    #print("", np.std(data))
    data /= 2.828#np.std(data)
    return data