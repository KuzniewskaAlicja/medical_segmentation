import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt


def create_data_mask_generator():
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1 / 255
    )

    mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1 / 255
    )

    return image_gen, mask_gen


def data_from_directory_gen(train_path, val_path, image_gen, mask_gen):
    flow_params = {'target_size': (256, 256),
                   'color_mode': "grayscale",
                   'class_mode': None,
                   'seed': 42,
                  }

    train_images = image_gen.flow_from_directory(
        f'{train_path}/data',
        batch_size=5,
        **flow_params
    ) 
    val_images = image_gen.flow_from_directory(
        f'{val_path}/data',
        batch_size=10,
        **flow_params
    )
    train_mask = mask_gen.flow_from_directory(
        f'{train_path}/mask',
        batch_size=5,
        **flow_params
    )
    val_mask = mask_gen.flow_from_directory(
        f'{val_path}/mask',
        batch_size=10,
        **flow_params
    )

    train_batch_size, val_batch_size = train_images.batch_size, val_images.batch_size
    train_samples, val_samples = train_images.n, val_images.n 

    return (zip(train_images, train_mask),
            zip(val_images, val_mask),
            train_batch_size, val_batch_size,
            train_samples, val_samples)
   
def check_first_five(gen):
    print(20 * '-' + 'Checking' + 20 * '-')
    images, masks = next(gen)

    for idx, (image, mask) in enumerate(zip(images[:5], masks[:5])):
        plt.subplot(5, 1, idx + 1)
        plt.imshow(np.squeeze(np.concatenate((image, mask), axis=1)), cmap = 'gray')
        plt.axis('off')
    plt.show()
