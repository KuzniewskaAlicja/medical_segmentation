import tensorflow as tf 
from tensorflow.keras import backend as K


class Unet():
    def __init__(self, input_size):
        self.inputs = tf.keras.layers.Input(input_size)
        self.conv_params = {'activation': 'relu',
                            'padding': 'same',
                            'kernel_initializer': 'he_normal'}

    def contraction_path(self):
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), **self.conv_params)(self.inputs)
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), **self.conv_params)(self.conv1)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv1)

        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), **self.conv_params)(pool1)
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), **self.conv_params)(self.conv2)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv2)

        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), **self.conv_params)(pool2)
        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), **self.conv_params)(self.conv3)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv3)

        self.conv4 = tf.keras.layers.Conv2D(512, (3, 3), **self.conv_params)(pool3)
        self.conv4 = tf.keras.layers.Conv2D(512, (3, 3), **self.conv_params)(self.conv4)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2))(self.conv4)

        self.conv5 = tf.keras.layers.Conv2D(1024, (3, 3), **self.conv_params)(pool4)
        self.conv5 = tf.keras.layers.Conv2D(1024, (3, 3), **self.conv_params)(self.conv5)

    def expansive_path(self):
        up6 = tf.keras.layers.UpSampling2D((2, 2))(self.conv5)
        up6 = tf.keras.layers.Conv2D(512, (2, 2), **self.conv_params)(up6)
        merge6 = tf.keras.layers.concatenate([self.conv4, up6])
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), **self.conv_params)(merge6)
        conv6 = tf.keras.layers.Conv2D(512, (3, 3), **self.conv_params)(conv6) 
        
        up7 = tf.keras.layers.UpSampling2D((2, 2))(conv6)
        up7 = tf.keras.layers.Conv2D(256, (2, 2), **self.conv_params)(up7)
        merge7 = tf.keras.layers.concatenate([self.conv3, up7])
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), **self.conv_params)(merge7)
        conv7 = tf.keras.layers.Conv2D(256, (3, 3), **self.conv_params)(conv7)

        up8 = tf.keras.layers.UpSampling2D((2, 2))(conv7)
        up8 = tf.keras.layers.Conv2D(128, (2, 2), **self.conv_params)(up8)
        merge8 = tf.keras.layers.concatenate([self.conv2, up8])
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), **self.conv_params)(merge8)
        conv8 = tf.keras.layers.Conv2D(128, (3, 3), **self.conv_params)(conv8)

        up9 = tf.keras.layers.UpSampling2D((2, 2))(conv8)
        up9 = tf.keras.layers.Conv2D(64, (2, 2), **self.conv_params)(up9)
        merge9 = tf.keras.layers.concatenate([self.conv1, up9])
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), **self.conv_params)(merge9)
        conv9 = tf.keras.layers.Conv2D(64, (3, 3), **self.conv_params)(conv9)

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
