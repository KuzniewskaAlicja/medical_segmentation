import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from scan_operations import load_raw_volume, load_labels_volume


def split_data(first_data_path, first_mask_path):
    train_data, val_data, train_label, val_label = \
    train_test_split(first_data_path, first_mask_path, 
                     test_size = 0.15, random_state = 42)
    
    return (train_data, val_data, 
            train_label, val_label)


def get_data_paths():
    first_dataset_path = Path('./scans/FirstDataset/train')
    second_dataset_path = Path('./scans/SecondDataset/train')

    # first dataset paths
    first_data_paths = [name for name in sorted(first_dataset_path.iterdir())
                        if not name.name.endswith('mask.nii.gz')]
    first_mask_paths = [name for name in sorted(first_dataset_path.iterdir())
                        if name.name.endswith('mask.nii.gz')]

    # second dataset paths
    second_data_paths = [Path(str(path / 'T1w.nii.gz'))
                         for path in sorted(second_dataset_path.iterdir())]
    second_mask_paths = [Path(str(path / 'mask.nii.gz'))
                         for path in sorted(second_dataset_path.iterdir())]

    return (first_data_paths, first_mask_paths,
            second_data_paths, second_mask_paths)


def final_save_path(path: Path, curr_save_path: Path,
                    mask: bool, first: bool, idx: int,
                    save_format: str) -> Path:
    if first and mask:
        save_path = Path(f'{curr_save_path}/{path.name.strip("_mask.nii.gz")}_{idx}.{save_format}')
    elif first and not mask:
        save_path = Path(f'{curr_save_path}/{path.name.strip(".nii.gz")}_{idx}.{save_format}')
    else:
        save_path = Path(f'{curr_save_path}/{path.parent.name}_{idx}.{save_format}')
    return save_path


def save_slices(path: Path, save_format: str='png', first: bool=True,
                train: bool=True, mask: bool=False):
    dataset_path = Path('./dataset/train' if train else './dataset/val')
    save_path = dataset_path / 'data/images' if not mask else dataset_path / 'mask/images'

    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)

    data = None
    if not mask:
        data, _ = load_raw_volume(path)
    else:
        data = load_labels_volume(path)

    if data is not None:
        for index in range(data.shape[0]):
            plt.imsave(final_save_path(path, save_path, mask,
                                       first, index, save_format),
                       data[index].T, cmap="gray", origin="lower",
                       format=save_format)


if __name__ == "__main__":
    first_data_path, first_mask_path, second_data_path, second_mask_path = get_data_paths()
    first_dataset_data = split_data(first_data_path, first_mask_path)
    second_dataset_data = split_data(second_data_path, second_mask_path) 
    
    save_params = {'first': [True] * 4 + [False] * 4,
                   'train': [True, False] * 4,
                   'mask': [False, False, True, True] * 2}
    
    for i, paths in enumerate((*first_dataset_data, *second_dataset_data,)):
        for idx, path in enumerate(paths):
            print(f'progress: {idx} / {len(paths)}')
            save_slices(path, first=save_params.get('first')[i],
                        train=save_params.get('train')[i],
                        mask=save_params.get('mask')[i])
            