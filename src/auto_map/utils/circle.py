import numpy as np

from functools import lru_cache
from utils.const_table import cos_phi, sin_phi


@lru_cache(maxsize=128)
def get_x_y(shape, rho):
    x_test = (rho * cos_phi + shape[1] // 2).astype(int)
    y_test = (rho * sin_phi + shape[0] // 2).astype(int)

    return x_test, y_test


def get_circle(img, rho):
    x_test, y_test = get_x_y(img.shape, rho)
    img_colors = img[y_test, x_test]

    res = np.column_stack((x_test, y_test, img_colors.reshape((720, 1))))
    return res
