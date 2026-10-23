import cv2
import numpy as np


def draw_rotate(img, center, coords_new, coords_result):
    backtorgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    R = int(0.7 * center)
    cv2.circle(backtorgb, (center, center), R, (0, 0, 255), 1)

    r = int(0.35 * center)
    cv2.circle(backtorgb, (center, center), r, (0, 0, 255), 1)

    for i in np.linspace(0.3, 0.9, 5):
        cv2.circle(backtorgb, (center, center), int(r*i), (255, 0, 0), 1)

    for sec in coords_new[::int(len(coords_new) / 16)]:
        cv2.line(backtorgb, (center, center), (int(sec[0]), int(sec[1])), (0, 255, 0), thickness=2)
    cv2.line(backtorgb, (center, center), (coords_new[0][0], coords_new[0][1]), (0, 0, 255), thickness=2)

    coords_cod_new = coords_result
    for sec in coords_cod_new[::int(len(coords_cod_new) / 16)]:
        cv2.line(backtorgb, (center, center), (int(sec[0]), int(sec[1])), (0, 255, 0), thickness=2)
    cv2.line(backtorgb, (center, center), (coords_cod_new[0][0], coords_cod_new[0][1]), (0, 0, 255), thickness=2)

    return backtorgb
