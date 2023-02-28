import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from keras import backend as K
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.utils import to_categorical
import cv2


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load info file
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/**ANATOMY_FUNCTION**/anatomy_function.csv')

# Loading images
def load_images(directory, im_size):
    # initialize our images array
    images = []
    indices = []
    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                # Loading images
                file_name = os.path.basename(file)[0]
                if file_name == 'b':
                    img1 = mpimg.imread(os.path.join(dir_path, file))
                    img1 = resize(img1, (im_size, im_size))
                if file_name == 'm':
                    img2 = mpimg.imread(os.path.join(dir_path, file))
                    img2 = resize(img2, (im_size, im_size))
                if file_name == 'a':
                    img3 = mpimg.imread(os.path.join(dir_path, file))
                    img3 = resize(img3, (im_size, im_size))

                    out = cv2.vconcat([img1, img2, img3])
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    out = gray[..., np.newaxis]

                    indices.append(int(folder))
                    images.append(out)

    return (np.array(images), indices)

images, indices = load_images(args["directory"], im_size=224)
images = images / 255.0
df = pd.DataFrame(indices, columns=['ID'])
info_df = pd.merge(df, patient_df, on=['ID'])


# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 672
BATCH_SIZE = 16
NUM_EPOCHS = 70
STEP_PER_EPOCH = 2
N_CLASSES = 15
CHECKPOINT_PATH = os.path.join("model_weights", "cp-{epoch:02d}")

#''' Fine tuning step '''

import ssl
from keras import Model, Sequential
from keras.layers import Dropout, Flatten, Dense

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from keras.applications.vgg16 import VGG16
model = VGG16(include_top=True, weights='imagenet')

transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)

for layer in vgg_model.layers[0:17]:
    layer.trainable = False
my_model = Sequential()
my_model.add(vgg_model)
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(2, activation='sigmoid'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])
