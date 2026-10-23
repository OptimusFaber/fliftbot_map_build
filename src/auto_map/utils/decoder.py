from pathlib import Path

import cv2
import numpy as np

from utils.circle import get_circle
from utils.draw_rotate import draw_rotate
from utils.hamming import decode


def polar_search_border(circle):
    borders = np.where(circle[1:, 2] != circle[:-1, 2])[0]
    if len(borders) == 0:
        return None, None
    g = borders[0] + 1
    return g, circle[g]


def get_code_polar_second(circle):
    borders = np.where(circle[1:, 2] != circle[:-1, 2])[0]
    if not len(borders):
        """ когда все одного цвета"""
        return None, None
    code = ''
    for n in range(len(borders) + 1):
        if n == len(borders):
            # TODO: обработка граничной ситуации (когда граница линии на нуле)
            seg = 720 - borders[-1]
            color_dot = -5
        elif n == 0:
            seg = borders[n] - 0
            color_dot = 5
        else:
            seg = borders[n] - borders[n - 1]
            color_dot = borders[n - 1] + 5

        arc = round(seg / 45)
        while arc != 0:
            color = circle[color_dot][2]
            if color == 255:
                code += '0'
            else:
                code += '1'
            arc -= 1

    return int(code, 2), code


def calculation_code_threshold_polar(size_segment: int):
    img = np.full(size_segment, 255, dtype=np.uint8)
    max_sum = img.sum()
    code_threshold = 0.5 * max_sum
    return code_threshold


def get_code_polar(line):
    one_segment = int(line.shape[0] / 16)

    code_threshold = calculation_code_threshold_polar(one_segment)

    code = ''
    for i in range(1, 16):
        start = i * one_segment
        stop = start + one_segment
        segment = line[start:stop, 2]
        segment_sum = segment.sum()
        if segment_sum > code_threshold:
            code += '0'
        else:
            code += '1'

    return int(code, 2), code


def find_rotation(img, center):
    clarification_borders_sum = 0
    cnt = 0
    res_boards = []
    for i in np.linspace(0.5, 0.7, 5):
        lines_new = get_circle(img, rho=int(i * center))

        res_bord, point = polar_search_border(lines_new)
        res_boards.append(res_bord)
        if res_bord is None:
            continue
        if cnt:
            if abs(res_bord - clarification_borders_sum / cnt) > 100:
                continue

        clarification_borders_sum += res_bord
        cnt += 1

    return int(clarification_borders_sum / cnt) if cnt else 0


def color_segment(coords_new, count_of_segments):
    return coords_new.reshape((count_of_segments, -1)).sum(axis=1)


def find_start_bit(img, len_segment, border_first, center, count_of_segments):
    r = int(0.35 * center)

    segments = np.zeros(count_of_segments)
    for i in np.linspace(0.3, 0.9, 5):
        rho = int(i * r)
        coords = get_circle(img, rho)

        coords_new = np.roll(coords, -border_first, axis=0)

        segments += color_segment(coords_new[:, 2], count_of_segments)

    index_max_first = np.argmax(segments)

    # return (index_max_first - 1) % 16 * len_segment

    segments[index_max_first] = 0
    index_max_two = np.argmax(segments)

    if index_max_first == count_of_segments - 1 and index_max_two == 0:
        return (count_of_segments - 1) * len_segment
    if index_max_first == 0 and index_max_two == count_of_segments - 1:
        return (count_of_segments - 1) * len_segment

    return np.min([index_max_first, index_max_two]) * len_segment


def find_start_bit_new(circle):

    borders = np.where(circle[1:, 2] != circle[:-1, 2])[0]

    for n in range(len(borders)):
        if n == len(borders) - 1:
            # TODO: обработка граничной ситуации (когда граница линии на нуле)
            seg = 720 - borders[-1] + borders[0]
            color_dot = 0
        else:
            seg = borders[n + 1] - borders[n]
            color_dot = borders[n] + 5
        if color_dot >= 720:
            color_dot -= 720
        color_first = circle[color_dot][2]
        color_second = circle[color_dot-10][2]

        if color_first == 255 and color_second != 255:
            return borders[n]


def find_opposite(img, value, centre, max_radius_of_search=0):
    for i in range(1, max_radius_of_search):
        slice = img[centre - i: centre + i, centre - i: centre + i]
        if not value in slice:
            continue
        coord = np.where(slice == value)
        return coord[0][0] - i, coord[1][0] - i
    return 0, 0


def find_centre_offset(img, centre):
    if img[centre, centre] == 0:
        return find_opposite(img, 255, centre)
    else:
        return find_opposite(img, 0, centre)


def img_decode_polar(img, draw_test=False, count_of_segments=16):
    if type(img) == str:
        img = cv2.imread(img, 0)
        # img = cv2.rotate(img, cv2.ROTATE_180)
        # img = img[9:,:]
        # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        # img = cv2.resize(img, (512, 512))
    center = int(img.shape[0] / 2)
    new_centre = find_centre_offset(img, center)
    # new_centre = 0, 0
    img = np.roll(img, -new_centre[1], axis=1)
    img = np.roll(img, -new_centre[0], axis=0)

    coords = get_circle(img, rho=int(0.7 * center))

    border_first = find_rotation(img, center)
    coords_new = np.roll(coords, -border_first, axis=0)

    # border_first = 0
    # coords_new = coords

    len_segment = coords_new.shape[0] // count_of_segments
    # border = find_start_bit(img, len_segment, border_first, center, count_of_segments)

    rho = int(0.3 * center)
    coords_r = get_circle(img, rho=rho)
    coords_new_r = np.roll(coords_r, -border_first, axis=0)

    border = find_start_bit_new(coords_new_r)
    if not border:
        border = find_start_bit(img, len_segment, border_first, center, count_of_segments)

    coords_result = np.roll(coords_new, -border, axis=0)

    if draw_test:
        img_with_rotation = draw_rotate(img, center, coords_new, coords_result)

    code, code_str = get_code_polar(coords_result)
    code_2, code_str_2 = get_code_polar_second(coords_result)
    if code_2:
        return decode(code_2)
    else:
        return -2


if __name__ == "__main__":
    res = img_decode_polar('/home/user/projects/navigation/errors/55/1_binary.png', draw_test=True)
    print(res)
    # decode(int("100110100010111", 2))
