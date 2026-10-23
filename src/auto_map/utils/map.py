import numpy.typing as npt
import numpy as np
import math as m
from numpy import cos, sin
from typing import Iterable


def calc_map(solved_lbls: set[int], angles_dict: dict, dist_dict: dict, coords: dict):
    labels2solve = find_lbls_to_solve(solved_lbls, angles_dict)
    while len(labels2solve):
        for lbl in labels2solve:
            neighbours = find_neighbours_to_solve(lbl, solved_lbls, angles_dict, dist_dict)
            estimated_coordinates = list()
            for neighbour in neighbours:
                calc_coord = find_coords(neighbour[0], neighbour[2], lbl, coords, angles_dict, dist_dict)
                estimated_coordinates.append(calc_coord)
            if len(estimated_coordinates):
                coords[lbl] = mean_coord(estimated_coordinates)
                solved_lbls.add(lbl)
        labels2solve = find_lbls_to_solve(solved_lbls, angles_dict)
        print("op")
    return coords


def find_lbls_to_solve(solved_lbls: set, angles_dict: dict) -> set:
    lables2solve = set()
    for angle_lbls in angles_dict.keys():
        angle_lbls: tuple[int, int, int]
        union = set(angle_lbls).intersection(solved_lbls)
        if len(union) == 2:
            lables2solve.update(set(angle_lbls).difference(union))
    return lables2solve


def find_neighbours_to_solve(lbl: int, solved_lbls: set, angles_dict: dict, dist_dict: dict):
    neighbours = list()
    for angle_lbls in angles_dict.keys():
        if lbl != angle_lbls[1]:
            continue
        other2calc = solved_lbls.intersection(angle_lbls)
        if len(other2calc) != 2:
            continue
        other2calc = tuple(other2calc)
        if (min(other2calc[0], lbl), max(other2calc[0], lbl)) not in dist_dict:
            continue
        if (min(other2calc[1], lbl), max(other2calc[1], lbl)) not in dist_dict:
            continue
        neighbours.append(angle_lbls)
    return neighbours


def find_coords(lbl1: int, lbl2: int, needed_lbl: int, coords: dict[int, npt.NDArray], angles: dict, lenghts: dict):
    d1 = np.array(coords[lbl1])
    d2 = np.array(coords[lbl2])
    len1_3 = lenghts[(min(lbl1, needed_lbl), max(lbl1, needed_lbl))]
    len1_3 = sum(len1_3) / len(len1_3)
    len2_3 = lenghts[(min(lbl2, needed_lbl), max(lbl2, needed_lbl))]
    len2_3 = sum(len2_3) / len(len2_3)
    a1_2_3 = get_angle(lbl1, lbl2, needed_lbl, angles)
    # a1_2_3 = sum(a1_2_3) / len(a1_2_3)  # - np.pi
    a2_1_3 = get_angle(lbl2, lbl1, needed_lbl, angles)
    # a2_1_3 = sum(a2_1_3) / len(a2_1_3)  # - np.pi

    vec12 = d1 - d2
    rot_matrix1 = np.array(((cos(a1_2_3), sin(a1_2_3)),
                           (-sin(a1_2_3), cos(a1_2_3))))
    vec23 = np.dot(rot_matrix1, vec12) * (len2_3 / np.linalg.norm(vec12))
    d3_1 = d2 + vec23

    vec21 = d2 - d1
    rot_matrix2 = np.array(((cos(a2_1_3), sin(a2_1_3)),
                           (-sin(a2_1_3), cos(a2_1_3))))
    vec13 = np.dot(rot_matrix2, vec21) * (len1_3 / np.linalg.norm(vec21))
    d3_2 = d1 + vec13
    # real_coords = coords[needed_lbl]
    return mean_coord((d3_1, d3_2))


def get_angle(lbl1: int, lbl2: int, lbl3: int, angles: dict):
    if (lbl1, lbl2, lbl3) in angles:
        a = sum(angles[(lbl1, lbl2, lbl3)]) / len(angles[(lbl1, lbl2, lbl3)])
        return a
    elif (lbl3, lbl2, lbl1) in angles:
        a = sum(angles[(lbl3, lbl2, lbl1)]) / len(angles[(lbl3, lbl2, lbl1)])
        return -a
    else:
        return None


def mean_coord(coords: Iterable[tuple[float, float]]):
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    return float(sum(xs)/len(xs)), float(sum(ys)/len(ys))
