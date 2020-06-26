import tensorflow as tf 
from tensorflow.keras import backend as K

class Unet():
    def __init__(self, input_size):
        self.inputs = tf.keras.layers.Input(input_size)
        self.conv_params = {'activation': 'relu',
                            'padding': 'same',
                            'kernel_initializer': 'he_normal'}

    def __double_conv(self, filters_nb, kernel_size: tuple, 
                      previous_layer: tf.keras.layers) -> tf.keras.layers:
        conv = tf.keras.layers.Conv2D(filters_nb, kernel_size,
                                      **self.conv_params)(previous_layer)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(filters_nb, kernel_size,
                                      **self.conv_params)(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)

        return conv

    def __up_sampling(self, filters_nb, up_size: tuple, conv_size: tuple,
                      previous_layer: tf.keras.layers, 
                      concatenate_layer: tf.keras.layers) -> tf.keras.layers:
        up = tf.keras.layers.UpSampling2D(up_size)(previous_layer)
        up = tf.keras.layers.Conv2D(filters_nb, conv_size,
                                    **self.conv_params)(up)
        merge = tf.keras.layers.concatenate([concatenate_layer, up])

        return merge

    def contraction_path(self):
        self.conv1 = self.__double_conv(64, (3, 3), self.inputs)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv1)

        self.conv2 = self.__double_conv(128, (3, 3), pool1)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv2)

        self.conv3 = self.__double_conv(256, (3, 3), pool2)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv3)

        self.conv4 = self.__double_conv(512, (3, 3), pool3)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv4)

        self.conv5 = self.__double_conv(1024, (3, 3), pool4)

    def expansive_path(self):
        up6 = self.__up_sampling(512, (2, 2), (2, 2), self.conv5, self.conv4)
        conv6 = self.__double_conv(512, (3, 3), up6)
        
        up7 = self.__up_sampling(256, (2, 2), (2, 2), conv6, self.conv3)
        conv7 = self.__double_conv(256, (3, 3), up7)

        up8 = self.__up_sampling(128, (2, 2), (2, 2), conv7, self.conv2)
        conv8 = self.__double_conv(128, (3, 3), up8)

        up9 = self.__up_sampling(64, (2, 2), (2, 2), conv8, self.conv1)
        conv9 = self.__double_conv(64, (3, 3), up9)

        self.outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    def Model(self):
        self.contraction_path()
        self.expansive_path()
        
        return self.inputs, self.outputs
        
    @staticmethod
    def dice_coef(y_true, y_pred):
        numerator = 2. * K.sum(K.abs(y_true * y_pred))
        denominator = K.sum(y_true + y_pred)
        return (numerator + 1) / (denominator + 1)

    @staticmethod
    def dice_loss(y_true, y_pred):
        return 1 - Unet.dice_coef(y_true, y_pred)
