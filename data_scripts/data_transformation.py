import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data: np.ndarray) -> np.ndarray:
    data = cv2.resize(data, (256, 256),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    
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


def rotate(array: np.ndarray, angle: int) -> np.ndarray:
    center = (array.shape[0] // 2, array.shape[1] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(array, rot_mat, (array.shape[1], array.shape[0]))
    
    return rotated
