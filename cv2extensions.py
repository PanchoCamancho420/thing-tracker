import numpy as np


def resize(img, size):
    x_size = img.shape[1]
    if x_size > size:
        sample_rate = int(np.ceil(float(x_size / float(size))))
        return img[0::sample_rate, 0::sample_rate]
    else:
        return img
