import numpy as np
import numpy.typing as npt
import cv2
import math as m
from pathlib import Path
import json
from utils.classes_no_torch import PTrack
from math import cos, sin, pi, sqrt
import pickle


def crops_from_dets(dets, img, offset_percent=0.1, square_thresh=0.9):
    assert square_thresh < 1 or square_thresh >= 0, "square_thresh must be 0 <square_thresh<1"
    imgs = []
    normal_dets = []
    real_crop = []
    for det in dets:
        ratio = (det[3] - det[1]) / (det[2] - det[0])
        off = (det[3] - det[1]) * offset_percent
        if not (square_thresh < ratio < 1 / square_thresh):
            continue
        if (det[0] - off < 3) or (det[1] - off < 3) or (det[3] + off > img.shape[0] - 3) or (
                det[2] + off > img.shape[1] - 3):
            continue
        imgs.append(img[int(det[1] - off): int(det[3] + off), int(det[0] - off): int(det[2] + off)])
        normal_dets.append(det)
        real_crop.append((int(det[1] - off), int(det[3] + off), int(det[0] - off), int(det[2] + off)))
    return imgs, normal_dets, real_crop


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image sides
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def get_binary_label(work_img, mask, tblr, size=510):
    if min(work_img.shape[:2]) < 16:
        return
    t, b, l, r = tblr
    gray = cv2.cvtColor(work_img, cv2.COLOR_RGB2GRAY)
    big_img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    big_blur = cv2.GaussianBlur(big_img, (5, 5), 0)
    _, big1 = cv2.threshold(big_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    masked_img = big1.copy()
    masked_img[~mask] = 255
    bool_array = (masked_img != 255)
    l_r = bool_array.any(axis=0)
    left = np.argmax(l_r)
    right = l_r.shape[0] - np.argmax(l_r[::-1])
    center_x = ((right + left) / 2) / size * (r - l) + l

    t_b = bool_array.any(axis=1)
    top = np.argmax(t_b)
    bot = t_b.shape[0] - np.argmax(t_b[::-1])
    center_y = ((bot + top) / 2) / size * (b - t) + t

    cropped = masked_img[top: bot, left:right]
    final = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_NEAREST)
    return final, (int(center_x), int(center_y))


def load_tracks(data):
    set_track_ids = set()
    for frame_data in data:
        set_track_ids.update(frame_data)
    ptracks = {int(i): PTrack([None for _ in range(PTrack.total_count)], track_id=int(i)) for i in set_track_ids}
    ptracks: dict[int, PTrack]
    for frame_num, frame_data in enumerate(data):
        for track_id in frame_data:
            ptracks[int(track_id)].frames_data[frame_num] = frame_data[track_id]
    return ptracks


"""Track filtering"""


def remove_short_tracks(tracks: dict[int, PTrack], frame_storage: list[list], min_len=15):
    to_remove = []
    for track in tracks:
        # remove all tracks that are solved with code -2, as we confidently don't know what they are
        if tracks[track].check_solved() and tracks[track].final_code == -2:
            to_remove.append(track)
            continue
        track_len = 0
        for frame_data in tracks[track].frames_data:
            if frame_data is not None:
                track_len += 1
        if track_len <= min_len:
            to_remove.append(track)
    for k in to_remove:
        tracks.pop(k)
    to_remove = set(to_remove)
    for num, frame2track in enumerate(frame_storage):
        frame_storage[num] = list(set(frame2track).difference(to_remove))
    return


def track_code_filtering(tracks: dict[int, PTrack], thresh: float = 0.8, best_over_second_thresh: float = 3):
    """ Перебирает каждый трек, и в случае если отношение самого встречающегося значения code к общему кол-ву
    значений превышает порог thresh, заменяет остальные значения на самое встречающееся"""
    for track in tracks:
        track = tracks[track]
        values = dict()
        total_count = 0
        for frame_data in track.frames_data:
            if frame_data is None:
                continue
            code = frame_data["code"]
            total_count += 1
            # if code == -2:
            #     continue
            if code not in values:
                values[code] = 1
            else:
                values[code] += 1
        sorted_dict = [(k, values[k]) for k in values]
        sorted_dict = sorted(sorted_dict, key=lambda x: x[1], reverse=True)
        if (len(sorted_dict) > 0 and sorted_dict[0][1] >= total_count * thresh) or (len(sorted_dict) > 1 and sorted_dict[0][1] >= sorted_dict[1][1] * best_over_second_thresh):
            most_probable_code = sorted_dict[0][0]
            for frame_data in track.frames_data:
                if frame_data is None:
                    continue
                frame_data["code"] = most_probable_code


"""DRAWING"""

GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# def draw_labels(online_targets: list[STrack], frame):
#     for track in online_targets:
#         cv2.rectangle(frame, (track.l, track.t), (track.r, track.b), (0, 255, 0), 2)
#         cv2.putText(frame, str(track.curr_code), (track.l, track.t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#         (0, 255, 0), 2)


def draw_labels(tracks: dict, frame: npt.NDArray):
    for track in tracks:
        track = tracks[track]
        t, l, b, r = track["tlbr"]
        t, l, b, r = map(int, (t, l, b, r))
        cv2.rectangle(frame, (l, t), (r, b), GREEN, 2)
        cv2.putText(frame, str(track["code"]), (l, t - 10), FONT, 0.5, GREEN, 2)


def draw_ptracks(tracks: list[PTrack], frame_num: int, frame: npt.NDArray):
    for track in tracks:
        t, l, b, r = track.frames_data[frame_num]['tlbr']
        t, l, b, r = map(int, (t, l, b, r))
        color = GREEN if track.is_solved else RED
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, str(track.frames_data[frame_num]['code']), (l, t - 10), FONT, 0.5, color, 2)


"""Working with files"""


def prepare_data_for_saving(tracks: dict[int, PTrack], frame2track: list[list[int]]):
    data = list()
    for num, tracks_nums in enumerate(frame2track):
        frame_data = dict()
        for track_num in tracks_nums:
            frame_data.update({track_num: tracks[track_num].frames_data[num]})
        data.append(frame_data)
    return data


def save_video_data(data: list[dict], video_path: str | Path, save_dir="output", overwrite: bool = False):
    video_path = Path(video_path)

    # Проверяем, существует ли директория save_dir, и создаем её, если нет
    save_dir_path = Path(save_dir)
    if not save_dir_path.exists():
        save_dir_path.mkdir(parents=True)

    # Определяем имя файла для сохранения данных
    base_name = video_path.stem
    file_extension = ".json"
    file_name = base_name + file_extension
    file_path = save_dir_path / file_name

    if overwrite:
        counter = 1
        while file_path.exists():
            file_name = f"{base_name}_{counter + 1}{file_extension}"
            file_path = save_dir_path / file_name
            if file_path.exists():
                counter += 1
            else:
                file_name = f"{base_name}_{counter}{file_extension}"
                file_path = save_dir_path / file_name
                break
    else:
        # Проверяем, существует ли файл с таким именем, и делаем инкремент, если необходимо
        counter = 1
        while file_path.exists():
            file_name = f"{base_name}_{counter}{file_extension}"
            file_path = save_dir_path / file_name
            counter += 1

    # Сохраняем данные в формате JSON
    data = {"video_path": str(video_path), "data": data}
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data saved to {file_path}")


def load_l2c_data(file_path: str | Path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    try:
        with open(file_path, 'rb') as f:
            l2c = pickle.load(f)
    except EOFError:
        print("Can't deserialize pickle file")
    return l2c


def load_video_data(file_path: str | Path) -> (Path, list[dict]):
    # Преобразуем file_path в объект Path, если это необходимо
    file_path = Path(file_path)

    # Проверяем, существует ли файл
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    # Читаем данные из файла JSON
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except OSError:
        return "Os error while opening json"
    except json.JSONDecodeError:
        return "Deserialized data is not a valid JSON document"
    except TypeError:
        return "The JSON object must be str, bytes or bytearray"
    error = check_data_correctness(data)
    if error:
        print(error)
        return error
    return data["video_path"], data["data"]


def check_data_correctness(data: dict, possible_coords_offset=1000):
    if type(data) is not dict:
        return "Bad json format"

    if ("video_path" not in data) or ("data" not in data):
        return "Bad json format"
    video_path = data["video_path"]
    data = data["data"]

    if type(video_path) is not str:
        return f"Video not found {video_path}"
    if not Path(video_path).exists():
        return f"fVideo not found {video_path}"
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise cv2.error("Unable to open video file")
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _, _ = cap.read()
    except cv2.error:
        return "Some cv2 error raised while video tries to read"
    except Exception:
        return "Video file is not correct"

    if type(data) is not list:
        return "Bad json format"
    if len(data) != frame_number:
        return f"Video contains {frame_number}, but json {len(data)}"

    for elem in data:
        if type(elem) is not dict:
            return "Bad json format"
        for label in elem:
            label = elem[label]
            if 'tlbr' not in label:
                return "Bad json format"

            has_single_code = 'code' in label
            has_raw_codes = ('codes' in label) and ('confs' in label)
            if not has_single_code and not has_raw_codes:
                return "Bad json format"

            if has_single_code:
                try:
                    int(label['code'])
                except (ValueError, TypeError):
                    return "Bad json format"

            if has_raw_codes:
                if type(label['codes']) is not list or type(label['confs']) is not list:
                    return "Bad json format"
                if len(label['codes']) != len(label['confs']):
                    return "Bad json format"
                try:
                    for code in label['codes']:
                        int(code)
                    for conf in label['confs']:
                        conf_float = float(conf)
                        if not (0.0 <= conf_float <= 1.0):
                            return "Bad json format"
                except (ValueError, TypeError):
                    return "Bad json format"

            if type(label['tlbr']) is not list:
                return "Bad json format"
            if len(label['tlbr']) != 4:
                return "Bad json format"
            try:
                for i in range(4):
                    float(label['tlbr'][i])
            except (ValueError, TypeError):
                return "Bad json format"

            off = possible_coords_offset
            if not (-off <= label['tlbr'][0] <= video_h+off):
                return "Coordinates are out of bounds"
            if not (-off <= label['tlbr'][2] <= video_h+off):
                return "Coordinates are out of bounds"
            if not (-off <= label['tlbr'][1] <= video_w+off):
                return "Coordinates are out of bounds"
            if not (-off <= label['tlbr'][3] <= video_w+off):
                return "Coordinates are out of bounds"


""" PART WITH MAP BUILDING"""


def coord_2_center(x, y, center_x=640, center_y=480):
    """ Cv2 корды к кордам от центра"""
    return x - center_x, -(y - center_y)


def coord_2_image(x, y, center_x=640, center_y=480):
    """ Корды посчитанные от центра в cv2 корды"""
    return x + center_x, center_y - y


def find_distance_homography_no_heights(track_data1, track_data2, h_matrix: npt.NDArray) -> float:
    point1 = np.dot(h_matrix, (*get_centre(track_data1), 1))  # type: np.ndarray
    point1 /= point1[2]
    point2 = np.dot(h_matrix, (*get_centre(track_data2), 1))  # type: np.ndarray
    point2 /= point2[2]
    return np.linalg.norm(point1 - point2)


def find_distance_homography_delme(track_data1, centre, h_matrix: npt.NDArray) -> float:
    point1 = np.dot(h_matrix, (*get_centre(track_data1), 1))  # type: np.ndarray
    point1 /= point1[2]
    point2 = np.dot(h_matrix, (int(centre[0]), int(centre[1]), 1))  # type: np.ndarray
    point2 /= point2[2]
    return np.linalg.norm(point1 - point2)


def find_distance_homography(track_data1, track_data2, h_matrix: npt.NDArray, heights: dict[int, float],
                             img_size: tuple[int, int], matrix_height: float) -> float:
    h1, h2 = heights[track_data1["code"]], heights[track_data2["code"]]
    w_img, h_img = img_size
    img_center_homo_cords = np.dot(h_matrix, (w_img // 2, h_img // 2, 1))  # type: np.ndarray
    img_center_homo_cords /= img_center_homo_cords[2]
    point1 = np.dot(h_matrix, (*get_centre(track_data1), 1))  # type: np.ndarray
    point1 /= point1[2]
    point2 = np.dot(h_matrix, (*get_centre(track_data2), 1))  # type: np.ndarray
    point2 /= point2[2]
    dist_center_1 = np.linalg.norm(point1 - img_center_homo_cords) * (h1 / matrix_height)
    dist_center_2 = np.linalg.norm(point2 - img_center_homo_cords) * (h2 / matrix_height)
    angle = find_angle_classic(get_centre(track_data1), (w_img // 2, h_img // 2), get_centre(track_data2))
    # gt = find_distance_homography_no_heights(track_data1, track_data2, h_matrix)
    # gt1 = find_distance_homography_delme(track_data1, (960, 540), h_matrix)
    # gt2 = find_distance_homography_delme(track_data2, (960, 540), h_matrix)
    calc_dist = sqrt(dist_center_1**2 + dist_center_2**2 - 2 * dist_center_1 * dist_center_2 * cos(angle-pi))
    return calc_dist


def find_proec(top_x: float, top_y: float, h_top: float, h_bot: float, distances_calib: npt.NDArray,
               ratio_kef: float, center_pix: tuple[int, int]) -> tuple[float, float]:
    """
    Пиксельные координаты метки проецирует так, если бы метка находилась ниже, на высоте h_bot.

    Args:
        top_x (): пиксельная координата x более высокой метки
        top_y (): пиксельная координата y более высокой метки
        h_top (): физическая высота высшей метки
        h_bot (): физическая высота низшей метки
        distances_calib (): матрица нахождения расстояний на камере
        ratio_kef (): коэффициент
        center_pix (): - центр изображения в пикселях

    Returns: Пиксельные координаты более высокой метки, если бы она физически находилась ниже на высоте h_bot.
    """
    len_0_2_real_mark = distances_calib[int(top_y), int(top_x)] * h_top * ratio_kef
    len_0_2_real_bot = distances_calib[int(top_y), int(top_x)] * h_bot * ratio_kef
    x_, y_ = coord_2_center(top_x, top_y, *center_pix)
    l_pix = (x_ ** 2 + y_ ** 2) ** 0.5
    l_pix_new = l_pix * len_0_2_real_mark / len_0_2_real_bot
    keef = l_pix_new / l_pix
    x_, y_ = x_ * keef, y_ * keef
    x_, y_ = coord_2_image(x_, y_, *center_pix)
    x2, y2 = int(x_), int(y_)
    return x2, y2


# RATIO_KEF = 0.032076
# RATIO_KEF = 0.0321  # 93.57
# RATIO_KEF = 0.0322 # 72.54
# RATIO_KEF = 0.03225  # 67.16
# RATIO_KEF = 0.03229  # 66.64
RATIO_KEF = 0.03228  # 66.43


def find_distance_classic(track_data1, track_data2, distances_calib: npt.NDArray, heights: dict[int, float],
                          center_pix: tuple[int, int], ratio_kef=RATIO_KEF) -> float:
    h1, h2 = heights[track_data1["code"]], heights[track_data2["code"]]
    y1, x1 = get_centre(track_data1)
    y2, x2 = get_centre(track_data2)
    height_ratio = min(h1, h2) * ratio_kef
    top_label = track_data1 if h1 >= h2 else track_data2
    bot_label = track_data1 if track_data1 != top_label else track_data2
    top_y, top_x = get_centre(top_label)
    x_proec, y_proec = find_proec(top_x, top_y, heights[top_label["code"]], heights[bot_label["code"]], distances_calib,
                                  ratio_kef, center_pix)
    if top_label == track_data1:
        x1, y1 = x_proec, y_proec
    else:
        x2, y2 = x_proec, y_proec

    ''' Вычисление косинуса между векторами в пикселях'''
    x1__, y1__ = coord_2_center(x1, y1, *center_pix)
    x2__, y2__ = coord_2_center(x2, y2, *center_pix)
    l2_pix = (x2__ ** 2 + y2__ ** 2) ** 0.5
    l1_pix = (x1__ ** 2 + y1__ ** 2) ** 0.5
    pix_cos = (x2__ * x1__ + y1__ * y2__) / (l2_pix * l1_pix)

    len_0_1_real = distances_calib[int(y1), int(x1)]
    len_0_1_real *= height_ratio
    len_0_2_real = distances_calib[int(y2), int(x2)]
    len_0_2_real *= height_ratio
    len_between_2_dots_calc = m.sqrt(len_0_2_real ** 2 + len_0_1_real ** 2 - 2 * len_0_2_real * len_0_1_real * pix_cos)
    return len_between_2_dots_calc


def process_frame_angles(frame_data: dict, angles_dct: dict):
    frame_data = list(frame_data.values())
    if len(frame_data) < 3:
        return
    for i in range(len(frame_data) - 2):
        for j in range(i + 1, len(frame_data) - 1):
            for k in range(j + 1, len(frame_data)):
                for _set in ((i, j, k), (k, i, j), (j, k, i)):
                    angle = find_angle(get_centre(frame_data[_set[0]]), get_centre(frame_data[_set[1]]),
                                       get_centre(frame_data[_set[2]]))
                    codes = (frame_data[_set[0]]["code"], frame_data[_set[1]]["code"], frame_data[_set[2]]["code"])
                    if codes not in angles_dct:
                        angles_dct[codes] = [angle]
                    else:
                        angles_dct[codes].append(angle)


def process_frame_distances(frame_data: dict, distances_dct: dict, homography_matrix: npt.NDArray,
                            heights: dict[int, float], homo_height: float, img_size: tuple[int, int],
                            distances_calib=None) -> None:
    """ Обрабатывает кадр, находит дистанции между каждой парой меток и заносит информацию в distances_dct.

    Args:

        frame_data (dict): Данные о кадре в виде словаря, ключами являются номера меток, значениями - словарь с двумя
         ключами: 'tlbr' и 'code'. Такой же тип словаря находится в изначальной дате.
        distances_dct (dict): Словарь, ключи которого, являются картежами с двумя элементами - номерами меток,
         значениями являются расстояния между эти двумя метками. Номера меток в картеже в ключе идут всегда по порядку
         возрастания.
        homography_matrix (npt.NDArray): Матрица гомографии для этой камеры
        heights (dict): Словарь в котором ключ это код метки, а значение - её высота.
        homo_height (float): Высота на которой делалась матрица гомографии.
        img_size (): Размер изображения: tuple[weight, height].
    Returns:
    """
    frame_data = list(frame_data.values())
    if len(frame_data) < 2:
        return
    for i in range(len(frame_data)):
        if i == len(frame_data) - 1:
            continue
        first_track_data = frame_data[i]
        for j in range(i + 1, len(frame_data)):
            second_track_data = frame_data[j]
            code1, code2 = first_track_data["code"], second_track_data["code"]
            # dst = find_distance_homography(first_track_data, second_track_data, homography_matrix, heights, img_size,
            #                                homo_height)
            dst = find_distance_classic(first_track_data, second_track_data, distances_calib, heights,
                                        (img_size[0] // 2, img_size[1] // 2))
            # dst = find_distance_homography_no_heights(first_track_data, second_track_data, homography_matrix)
            dst = float(dst)
            key = (min(code1, code2), max(code1, code2))
            if key in distances_dct:
                distances_dct[key].append(dst)
            else:
                distances_dct[key] = [dst]


def find_angle(A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    # Векторы BA и BC
    BA = (x1 - x2, y1 - y2)
    BC = (x3 - x2, y3 - y2)

    # Вычисление угла между векторами BA и BC с учетом направления поворота
    angle = m.atan2(BC[1], BC[0]) - m.atan2(BA[1], BA[0])

    # Приведение угла к диапазону [-π, π]
    if angle > m.pi:
        angle -= 2 * m.pi
    elif angle < -m.pi:
        angle += 2 * m.pi

    return angle


#
def find_angle_classic(A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    AB = (x2 - x1, y2 - y1)
    BC = (x3 - x2, y3 - y2)
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]
    len_AB = m.sqrt(AB[0] ** 2 + AB[1] ** 2)
    len_BC = m.sqrt(BC[0] ** 2 + BC[1] ** 2)
    cos_theta = dot_product / (len_AB * len_BC)
    theta = m.acos(cos_theta)
    sign = AB[0] * BC[1] - AB[1] * BC[0]
    if sign < 0:
        theta = 2 * m.pi - theta
    return theta


def get_centre(data: dict):
    t, l, b, r = data["tlbr"]
    return (t + b) / 2, (l + r) / 2


def make_pix_proec(track_data, heights: dict[int: float], img_size: tuple[int, int], basic_height: float):
    h1 = heights[track_data["code"]]
    w_img, h_img = img_size
    x, y = get_centre(track_data)
    x_c, y_c = coord_2_center(x, y, w_img//2, h_img//2)
    length = np.linalg.norm(np.array(x_c, y_c))
    needed_lenght = length * (h1/basic_height)
    x_c_new, y_c_new = x_c/length * needed_lenght, y_c/length * needed_lenght
    return coord_2_image(x_c_new, y_c_new, w_img//2, h_img//2)


def save_angles(angles: dict[tuple[int, int], list]):
    new_angles = dict()
    for a in angles:
        res_angle = sum(angles[a]) / len(angles[a])
        new_angles[a] = res_angle
    with open("angles.pickle", 'wb') as file:
        pickle.dump(new_angles, file)
    return new_angles


def save_angles_and_dists(angles: dict[tuple[int, int], list], dists: dict[tuple[int, int], list]):
    angles = save_angles(angles)
    dists = save_dists(dists)
    return angles, dists


def save_dists(dists: dict[tuple[int, int], list]):
    new_dists = dict()
    for dist in dists:
        res_dist = sum(dists[dist]) / len(dists[dist])
        new_dists[dist] = res_dist
    with open("dists.pickle", 'wb') as file:
        pickle.dump(new_dists, file)
    return new_dists
