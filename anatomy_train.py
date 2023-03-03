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
patient_df = pd.read_csv('/Users/ebrahamalskaf/Documents/**ANATOMY_FUNCTION**/info.csv')

# Loading images
def load_images(directory, df, im_size):
    # initialize our images array
    images = []
    labels = []
    # Loop over folders and files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Collect perfusion .png images
        if len(files) > 1:
            folder = os.path.split(root)[1]
            dir_path = os.path.join(directory, folder)
            for file in files:
                if '.DS_Store' in files:
                    files.remove('.DS_Store')
                if int(folder) not in df['ID'].values:
                    continue
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

                    # Defining labels
                    if df[df["ID"].values == int(folder)]['MVD'].values == 1:
                        the_class = 1
                    elif df[df["ID"].values == int(folder)]['multi_ves'].values == 1:
                        the_class = 2
                    else:
                        the_class = 3

                    images.append(out)
                    labels.append(the_class)

    return (np.array(images), np.array(labels))

(images, labels) = load_images(args["directory"], patient_df, im_size=224)
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 3)

# Set parameters
INPUT_DIM = 224
WIDTH = 224
HEIGHT = 672
BATCH_SIZE = 16
NUM_EPOCHS = 250
STEP_PER_EPOCH = 2
N_CLASSES = 3
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

# Splitting data
(X_train, X_valid, y_train, y_valid) = train_test_split(images, labels, train_size=0.8, random_state=42)

# Data augmentation
aug = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

v_aug = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)

# Initialise the optimiser and model
print("[INFO] compiling model ...")
Opt = Adam(lr=0.001)
Loss = CategoricalCrossentropy(from_logits=True)
fa_model = tf_cnns.LeNet.build(WIDTH, HEIGHT, depth=1, classes=N_CLASSES)
fa_model.compile(loss=Loss, optimizer=Opt, metrics=["accuracy"])
weigth_path = "{}_my_model.best.hdf5".format("#fa_predictor1")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
early = EarlyStopping(monitor='val_loss', mode='min', patience=20)
callbacks_list = [checkpoint]

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
print("[INFO] Training the model ...")
history = fa_model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data= v_aug.flow(X_valid, y_valid), epochs=NUM_EPOCHS,
                  steps_per_epoch=len(X_train )// 16, callbacks=[callbacks_list, tensorboard_callback], verbose=1)

# summarize history for loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mortality CNN training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'], loc='upper left')
plt.show()

# Saving model data
model_json = fa_model.to_json()
with open("fa_model.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
