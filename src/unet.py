from keras.models import Model
from keras.layers import Input, MaxPooling2D, concatenate, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


def double_conv_layer(x, size, dropout, batch_norm):
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv


def get_unet(dropout_val=0.05, batch_norm=True, input_size=(224,224)):
    inputs = Input((input_size[0], input_size[1], INPUT_CHANNELS))
    conv1 = double_conv_layer(inputs, 32, dropout_val, batch_norm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64, dropout_val, batch_norm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128, dropout_val, batch_norm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256, dropout_val, batch_norm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512, dropout_val, batch_norm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = double_conv_layer(pool5, 1024, dropout_val, batch_norm)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=3)
    conv7 = double_conv_layer(up6, 512, dropout_val, batch_norm)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=3)
    conv8 = double_conv_layer(up7, 256, dropout_val, batch_norm)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=3)
    conv9 = double_conv_layer(up8, 128, dropout_val, batch_norm)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=3)
    conv10 = double_conv_layer(up9, 64, dropout_val, batch_norm)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=3)
    conv11 = double_conv_layer(up10, 32, 0, batch_norm)

    conv12 = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(conv11)
    conv12 = BatchNormalization(axis=3)(conv12)
    conv12 = Activation('sigmoid')(conv12)

    model = Model(input=inputs, outputs=conv12)
    return model