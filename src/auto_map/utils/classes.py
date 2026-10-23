import numpy as np
from ultralytics import YOLO
from utils.utils import crops_from_dets, create_circular_mask, get_binary_label
from utils.decoder import img_decode_polar
import sys
sys.path.append('/home/alricofrahls/bigdisk/AlricOfRahls/VkusvillODM/')
from fourier_decoder.decoder import FourierDecoder
import line_profiler
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, weights):
        YOLO(weights)


class Decoder:
    def __init__(self, labels_file: str):
        self.size_binary_label = 256
        self.crop_radius_binary_img = 0.9
        self.decoder = FourierDecoder.from_labels2coords(labels_file, top_n=1, tmp_dir="/home/alricofrahls/bigdisk/AlricOfRahls/VkusvillODM/auto_map/tmp/")

    @line_profiler.profile
    def prepare_and_decode(self, keypoints, im, coordinates=None):
        crops, keypoints, coords_of_crops = crops_from_dets(keypoints, im, offset_percent=0)
        # imshow the crops with plt
        # for i, crop in enumerate(crops):
        #     print(crop.shape)
        #     plt.imshow(crop)
        #     plt.title(f'crop_{i}')
        #     plt.show()
        # print(crops, crops[0].shape)
        # labels, scores = self.decoder.predict_with_labels(crops, point=coordinates, radius=2000.)
        labels, scores = self.decoder.predict_with_labels(crops)
        # print(labels, scores)
        # coordinates_new = self.decoder.get_average_position([label[0] for label in labels])
        coordinates_new = None
        # good_crops = list(
        #     map(lambda x: get_binary_label(x[0], self.circular_mask, x[1], size=self.size_binary_label),
        #         zip(crops, coords_of_crops)))

        res, kp2 = [], []
        for i in range(len(keypoints)):
            # if good_crops[i] is not None:
            code = labels[i][0]
            kp2.append(keypoints[i])
            res.append(code)
        return res, np.array(kp2), coordinates_new
