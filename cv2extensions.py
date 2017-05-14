import numpy as np


def resize(img, size):
    x_size = img.shape[1]
    if x_size > size:
        sample_rate = int(np.ceil(float(x_size / float(size))))
        return img[0::sample_rate, 0::sample_rate]
    else:
        return img


def rect(img, p1, p2, color=list((0, 255, 0))):
    img[p1[1], p1[0]] = color
    img[p2[1], p2[0]] = color

    for i in xrange(p1[1], p2[1]):
        img[i, p1[0]] = color
        img[i, p2[0]] = color

    for i in xrange(p1[0], p2[0]):
        img[p1[1], i] = color
        img[p2[1], i] = color
