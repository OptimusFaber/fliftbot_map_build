import pickle
import numpy as np
import math as m
from pathlib import Path
from utils.utils import find_angle


def show_angle_stats(angle_dict: dict, l2c_pth: str | Path):
    with open(l2c_pth, 'rb') as f:
        l2c = pickle.load(f)

    report = list()
    std_angle_sum = 0
    total_err = 0
    for angle in angle_dict:
        values = np.array(angle_dict[angle])
        std = std_angle(values)
        std_angle_sum += std
        angle_real = (2 * np.pi - find_angle(l2c[angle[0]], l2c[angle[1]], l2c[angle[2]]))
        err = angle_difference(mean_angle(values), angle_real)
        total_err += err
        report.append((angle, m.degrees(std), m.degrees(np.min(values)), m.degrees(np.max(values)), m.degrees(angle_real),
                       m.degrees(err)))
                      # m.degrees(abs((mean_angle(values) - (2 * m.pi - angle_real)) // (2*m.pi)))))
    report.sort(key=lambda x: x[5], reverse=True)
    for dat in report:
        print(f"{dat[0]}: стандартное отклонение (град): {dat[1]:.2f},"
              f"min {dat[2]:.2f}, max {dat[3]:.2f}, real {dat[4]:.2f}, err {dat[5]:.2f}")
    print(f"\033[94mTOTAL STD DEGREES\033[0m:\n   \033[92m{m.degrees(std_angle_sum)}\033[0m")
    print(f"\033[94mTOTAL ERR DEGREES\033[0m:\n   \033[92m{m.degrees(total_err)}\033[0m")


def compare_dist_with_real(distance_dict: dict, l2c_pth: str | Path):
    errors = []
    with open(l2c_pth, 'rb') as f:
        l2c = pickle.load(f)  # label2coords_file
    for pare in distance_dict:
        calc_dists = distance_dict[pare]
        calc_dist = sum(calc_dists) / len(calc_dists)  # TODO если среднего будет недостаточно, придумать другие методы
        # и через флаг здесь их использовать
        if pare[0] not in l2c or pare[1] not in l2c:
            continue  # этот иф добавил потом, когда хочу проверить на карте на которой не все метки размечены
        real_dist = np.linalg.norm(np.array(l2c[pare[0]]) - np.array(l2c[pare[1]]))
        # print(f"{pare}: {abs(calc_dist - real_dist)}")
        errors.append((pare, abs(calc_dist - real_dist)))
        errors.sort(key=lambda x: x[1], reverse=True)
    total_err = 0
    for error in errors:
        total_err += error[1]
        print(f"{error[0]}: {error[1]}")
    print(f"\033[94mTOTAL SCORE\033[0m (в попугаях, чем меньше тем круче):\n   \033[92m{total_err}\033[0m")


def mean_angle(angles):
    # Преобразование углов в комплексные числа
    complex_angles = np.exp(1j * angles)
    # Вычисление среднего комплексного числа
    mean_complex = np.mean(complex_angles)
    # Преобразование среднего комплексного числа обратно в угол
    _mean_angle = np.angle(mean_complex)
    return _mean_angle


def variance_angle(angles):
    # Преобразование углов в комплексные числа
    complex_angles = np.exp(1j * angles)
    # Вычисление среднего комплексного числа
    mean_complex = np.mean(complex_angles)
    # Вычисление дисперсии комплексных чисел
    variance = 1 - np.abs(mean_complex)
    return variance


def angle_difference(angle1, angle2):
    diff = angle2 - angle1
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff)


def std_angle(angles):
    # Вычисление дисперсии углов
    variance = variance_angle(angles)
    # Вычисление стандартного отклонения углов
    std_dev = np.sqrt(-2 * np.log(1 - variance))
    return std_dev
