import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import data_scripts.scan_operations as scan
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from network.unet import Unet


def rotate(array: np.ndarray, angle: int) -> np.ndarray:
    centre = (array.shape[0] // 2, array.shape[1] // 2)
    rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(array, rot_mat, (array.shape[1], array.shape[0]))
    
    return rotated


def preprocess_data(data: np.ndarray) -> np.ndarray:
    data = cv2.resize(data, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    data = rotate(data, 90)
    data = MinMaxScaler().fit_transform(data)
    data = np.expand_dims(data, axis=2)
    
    return data


def postprocess_data(input_data_size: tuple, data: np.ndarray) -> np.ndarray:
    data = data.squeeze()
    data = rotate(data, -90)
    data = np.where(data > 0.5, 1, 0)
    data = cv2.resize(data.astype('uint8'), input_data_size,
                      interpolation=cv2.INTER_AREA)
    return data


def save_labels(path: Path, model, first=True):
    save_dir, save_path = '', ''
    if first:
        save_dir = Path('./Predictions/FirstDataset')
    else:
        save_dir = Path('./Predictions/SecondDataset')

    for path_name in path.iterdir():
        data, affine = scan.load_raw_volume(path_name if first else Path(path_name / 'T1w.nii.gz'))
        labels = np.zeros(data.shape, dtype=np.uint8)

        if not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        
        if first:
            save_path = save_dir / path_name.name
        else:
            save_path = save_dir / f'{path_name.name}.nii.gz'

        x_size, y_size, z_size = data.shape
        for idx in range(x_size):
            input_data = preprocess_data(data[idx])
            prediction = model.predict(input_data[None, :])
            output = postprocess_data((z_size, y_size), prediction)
            labels[idx] = output
        scan.save_labels(labels, affine, save_path)

def main():
    first_dataset_test_path = Path('./scans/FirstDataset/test')
    second_dataset_test_path = Path('./scans/SecondDataset/test')

    custom_objects = {'dice_loss': Unet.dice_loss,
                      'dice_coef': Unet.dice_coef}

    model_path = './best_model/model_v006_0.98.h5'
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    save_labels(first_dataset_test_path, model, first=True)
    save_labels(second_dataset_test_path, model, first=False)


if __name__ == "__main__":
    main()
