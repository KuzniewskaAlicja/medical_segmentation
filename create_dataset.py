from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from scan_operations import load_raw_volume, load_labels_volume


def split_data(first_data_path, first_mask_path, second_data_path, second_mask_path):
    train_data, val_data, train_label, val_label = \
    train_test_split(first_data_path, first_mask_path, 
                     test_size = 0.15, random_state = 42)

    second_train_data, second_val_data, second_train_label, second_val_label = \
    train_test_split(second_data_path, second_mask_path,
                     test_size = 0.15, random_state=42)
    
    return (train_data, val_data, 
            train_label, val_label,
            second_train_data, second_val_data,
            second_train_label, second_val_label)


def get_data_paths():
    first_dataset_path = Path('./raw_data/FirstDataset/train')
    second_dataset_path = Path('./raw_data/SecondDataset/train')

    # first dataset paths
    first_data_paths = [name for name in sorted(first_dataset_path.iterdir()) if not name.name.endswith('mask.nii.gz')]
    first_mask_paths = [name for name in sorted(first_dataset_path.iterdir()) if name.name.endswith('mask.nii.gz')]

    # second dataset paths
    second_data_paths = [Path(str(path / 'T1w.nii.gz')) for path in sorted(second_dataset_path.iterdir())]
    second_mask_paths = [Path(str(path / 'mask.nii.gz')) for path in sorted(second_dataset_path.iterdir())]

    return (first_data_paths, first_mask_paths,
            second_data_paths, second_mask_paths)


def final_save_path(path, curr_save_path, mask, first, idx) -> Path:
    save_path = ''
    if first and mask:
        save_path = Path(f'{curr_save_path}/{path.name.strip("_mask.nii.gz")}_{idx}.png')
    elif first and not mask:
        save_path = Path(f'{curr_save_path}/{path.name.strip(".nii.gz")}_{idx}.png')
    else:
        save_path = Path(f'{curr_save_path}/{path.parent.name}_{idx}.png')
  
    return save_path


def save_slices(path, save_format='png', first=True, train=True, mask=False):
    save_path = Path('./dataset')
    if first:
        save_path = save_path / 'FirstDataset/train' if train else save_path / 'FirstDataset/val'
    else:
        save_path = save_path / 'SecondDataset/train' if train else save_path / 'SecondDataset/val'
  
    save_path = save_path / 'mask' if mask else save_path / 'data'
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)

    data = None
    if not mask:
        data, _ = load_raw_volume(path)
    else:
        data = load_labels_volume(path)

    if data is not None:
        for index in range(data.shape[0]):
            plt.imsave(final_save_path(path, save_path, mask, first, index), data[index].T, cmap="gray", origin="lower", format=save_format)


if __name__ == "__main__":
    first_data_path, first_mask_path, second_data_path, second_mask_path = get_data_paths()
    data = split_data(first_data_path, first_mask_path, second_data_path, second_mask_path) 
    
    params = {'first': [True] * 4 + [False] * 4,
              'train': [True, False] * 4,
              'mask': [False, False, True, True] * 2}

    for i, paths in enumerate(data):
        for idx, path in enumerate(paths):
            print(f'progress: {idx} / {len(paths)}')
            save_slices(path, first=params.get('first')[i],
                        train=params.get('train')[i],
                        mask=params.get('mask')[i])
            