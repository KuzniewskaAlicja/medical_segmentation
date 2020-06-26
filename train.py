from os import path, mkdir
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from tensorflow.keras.callbacks import ModelCheckpoint

from network.unet import Unet
import data_scripts.dataset_generator as dg


def data_generator(dataset_dir, image_gen, mask_gen, display_check=False):
    train_dir = f'{dataset_dir}/train'
    val_dir = f'{dataset_dir}/val'

    data_gen = dg.data_from_directory_gen(train_dir, val_dir,
                                          image_gen, mask_gen)

    if display_check:
        dg.check_first_five(data_gen[0])
        dg.check_first_five(data_gen[1])
    
    return data_gen

def save_model(dir_name, counter, model):
    model_name = f'model_v{counter:03}'

    if not path.exists(dir_name):
        mkdir(dir_name)

    model.save(f'{dir_name}/{model_name}.h5')
    

def main():
    dataset_dir = './dataset'
    DATA_SIZE = (256, 256, 1)
    counter = 1
    save_dir = './models'
    image_gen, mask_gen = dg.create_data_mask_generator()
    data_gen = data_generator(dataset_dir, image_gen, mask_gen, display_check=False)
    train_gen, val_gen = data_gen[:2]
    train_batch_size, val_batch_size, train_samples, val_samples = data_gen[2:]
    train = (pair for pair in train_gen)
    val = (pair for pair in val_gen)
    
    tf.keras.backend.clear_session()
    checkpoint = ModelCheckpoint(filepath='./models/model_v{epoch:03d}_{val_dice_coef:.2f}.h5',
                                 save_best_only=True,
                                 monitor='val_dice_coef',
                                 mode='max',
                                 verbose=1)
                                 
    inputs, outputs = Unet(DATA_SIZE).Model()
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss=Unet.dice_loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=[Unet.dice_coef])
    _ = model.fit_generator(train,
                            steps_per_epoch=train_samples // train_batch_size,
                            epochs=10,
                            validation_data=val,
                            validation_steps=val_samples // val_batch_size,
                            verbose=1,
                            use_multiprocessing=True, workers=4,
                            callbacks=[checkpoint])

    choice = input('Do you want to save this model?')
    if choice:
        save_model(save_dir, counter, model)
    
if __name__ == '__main__':
    main()
