import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, Flatten, AveragePooling2D, GlobalAveragePooling2D, Activation, ZeroPadding2D
from keras.layers.merge import concatenate, add
import matplotlib.pyplot as plt
import os
import time


''' Building AlexNet '''

def AlexNet(INPUT_SHAPE, OUTPUT):   
    model = Sequential([
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4)
                            , activation='relu', input_shape=INPUT_SHAPE),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),
                            activation='relu',padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
                            activation='relu',padding="same"),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
                            activation='relu',padding="same"),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
                            activation='relu',padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dense(0.5),
        Dense(OUTPUT, activation='softmax')
        ])
    return model


''' Building GoogleNet '''

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
  # Input:
  # - f1: number of filters of the 1x1 convolutional layer in the first path
  # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
  # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
  # - f4: number of filters of the 1x1 convolutional layer in the fourth path

  # 1st path:
  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  # 2nd path
  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  # 3rd path
  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  # 4th path
  path4 = MaxPool2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer


def GoogLeNet(INPUT_SHAPE, OUTPUT):
    # input layer
    input_layer = Input(INPUT_SHAPE)

    # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # convolutional layer: filters = 64, strides = 1
    X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(X)

    # convolutional layer: filters = 192, kernel_size = (3,3)
    X = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(X)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # 1st Inception block
    X = Inception_block(X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)

    # 2nd Inception block
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # 3rd Inception block
    X = Inception_block(X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)

    # Extra network 1:
    X1 = AveragePooling2D(pool_size=(5, 5), strides=3)(X)
    X1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(X1)
    X1 = Flatten()(X1)
    X1 = Dense(1024, activation='relu')(X1)
    X1 = Dropout(0.7)(X1)
    X1 = Dense(5, activation='softmax')(X1)

    # 4th Inception block
    X = Inception_block(X, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)

    # 5th Inception block
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)

    # 6th Inception block
    X = Inception_block(X, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)

    # Extra network 2:
    X2 = AveragePooling2D(pool_size=(5, 5), strides=3)(X)
    X2 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(X2)
    X2 = Flatten()(X2)
    X2 = Dense(1024, activation='relu')(X2)
    X2 = Dropout(0.7)(X2)
    X2 = Dense(1000, activation='softmax')(X2)

    # 7th Inception block
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32,
                        f3_conv5=128, f4=128)

    # max-pooling layer: pool_size = (3,3), strides = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2)(X)

    # 8th Inception block
    X = Inception_block(X, f1=256, f2_conv1=160, f2_conv3=320, f3_conv1=32, f3_conv5=128, f4=128)

    # 9th Inception block
    X = Inception_block(X, f1=384, f2_conv1=192, f2_conv3=384, f3_conv1=48, f3_conv5=128, f4=128)

    # Global Average pooling layer
    X = GlobalAveragePooling2D(name='GAPL')(X)

    # Dropoutlayer
    X = Dropout(0.4)(X)

    # output layer
    X = Dense(OUTPUT, activation='softmax')(X)

    # model
    model = Model(input_layer, [X, X1, X2], name='GoogLeNet')

    return model


''' Building ResNet '''

# A single resnet module consisting of 1 x 1 conv - 3 x 3 conv and 1 x 1 conv
def resnet_identity_module(x, filters, pool=False):
    res = x
    stride = 1
    if pool:
        stride = 2
        res = Conv2D(filters, kernel_size=1, strides=2, padding="same")(res)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(int(filters / 4), kernel_size=1, strides=stride, padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(int(filters / 4), kernel_size=3, strides=1, padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size=1, strides=1, padding="same")(x)

    x = add([x, res])


    return x

def resnet_first_identity_module(x, filters):
    res = x
    res = Conv2D(filters, kernel_size=1, strides=1, padding="same")(res)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(int(filters / 4), kernel_size=1, strides=1, padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(int(filters / 4), kernel_size=3, strides=1, padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size=1, strides=1, padding="same")(x)

    x = add([x, res])


    return x

def resnet_block(x, filters, num_layers, pool_first_layer=True):
    for i in range(num_layers):
        pool = False
        if i == 0 and pool_first_layer: pool = True
        x = resnet_identity_module(x, filters=filters, pool=pool)
    return x


def ResNet(INPUT_SHAPE, num_layers=50, OUTPUT=6):
    if num_layers not in [50, 101, 152]:
        raise ValueError("Num Layers must be either 50, 101 or 152")

    block_layers = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    block_filters = {
        50: [256, 512, 1024, 2048],
        101: [256, 512, 1024, 2048],
        152: [256, 512, 1024, 2048]
    }

    layers = block_layers[num_layers]
    filters = block_filters[num_layers]
    input = Input(INPUT_SHAPE)

    # Since the first layers in the modules are bn and relu, we do not include bn and relu after the first conv
    x = Conv2D(64, kernel_size=7, strides=2, padding="same")(input)

    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = resnet_first_identity_module(x, filters[0])

    for i in range(4):
        num_filters = filters[i]
        num_layers = layers[i]

        pool_first = True
        if i == 0:
            pool_first = False
            num_layers = num_layers - 1
        x = resnet_block(x, filters=num_filters,
                         num_layers=num_layers, pool_first_layer=pool_first)

    # Since the output of the residual unit is addition of convs, we need to appy bn and relu before global average pooling
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(OUTPUT)(x)
    x = Activation("softmax")(x)

    model = Model(inputs=input, outputs=x,
                  name="ResNet{}".format(num_layers))

    return model


''' Building VGGNet'''

def VGGNet(INPUT_SHAPE, OUTPUT):
    model = Sequential()
    model.add(Conv2D(input_shape=INPUT_SHAPE, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(OUTPUT, activation="softmax"))

    return model
