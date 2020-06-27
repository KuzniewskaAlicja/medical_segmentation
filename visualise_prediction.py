import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import data_scripts.scan_operations as scan


def normalize_data(data: np.ndarray) -> np.ndarray:
    return np.array(255 * data/max(data.ravel()), dtype=np.uint8)


def get_slice(data: np.ndarray, slice_nb: int) -> np.ndarray:
    return data[slice_nb]


def overlaying_images(data_slice: np.ndarray,
                      mask_slice: np.ndarray) -> np.ndarray:
    data_slice = cv2.cvtColor(data_slice, cv2.COLOR_GRAY2RGB)
    mask = np.zeros((*mask_slice.shape, 3), np.uint8)
    mask[mask_slice == 255] = (0, 0, 153)

    return cv2.addWeighted(data_slice, 0.5, mask, 0.5, 0)


def main():
    scan_path = './1c3064790751e45d903399e9998af51b.nii.gz'
    mask_path = './1c3064790751e45d903399e9998af51b_mask.nii.gz'
    data, _ = scan.load_raw_volume(Path(scan_path))
    mask = scan.load_labels_volume(Path(mask_path))

    example_samples = [50, 100, 150]
    for sample in example_samples:
        data_slice = normalize_data(get_slice(data, sample))
        mask_slice = get_slice(mask * 255, sample)
        result = overlaying_images(data_slice, mask_slice)

        if sample == example_samples[0]:
            output = result
        else:
            output = np.hstack([output, result])

    cv2.imwrite('./visualisation.jpg', output)
    cv2.imshow('result', output)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
