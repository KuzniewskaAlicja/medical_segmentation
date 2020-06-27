import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from data_scripts.data_transformation import preprocess_data, postprocess_data
import data_scripts.scan_operations as scan
from network.unet import Unet


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
