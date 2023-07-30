import keras.callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers.schedules import learning_rate_schedule
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tf_cnns
import utils
import argparse
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import class_weight
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
patient_df['epi_cad'] =  patient_df[['LMS','LAD','LCx','RCA']].apply(lambda x:'{}'.format(np.max(x)),axis=1)

(images, labels) = utils.load_perf_data(args["directory"], patient_df, im_size=224)
print(images.shape)
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

model = VGG16(include_top=False, input_shape=(HEIGHT, WIDTH, 3), weights='imagenet')
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
my_model.add(Dense(3, activation='softmax'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration
                                                                      (memory_limit=4096)])

# Splitting data
(X_train, X_valid, y_train, y_valid) = train_test_split(images, labels, train_size=0.8, stratify=labels, shuffle=True)

# Data augmentation
aug = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range
                         =0.2, horizontal_flip=True, fill_mode="nearest")

v_aug = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)

# Initialise the optimiser and model
print("[INFO] compiling model ...")
def scheduler(epoch, lr):
    if epoch <10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
Opt = Adam(lr=0.001)
Loss = CategoricalCrossentropy(from_logits=True)
my_model.compile(loss=Loss, optimizer=Opt, metrics=["accuracy"])
weigth_path = "{}_my_model.best.hdf5".format("models/mvd_epi_Res")
checkpoint = ModelCheckpoint(weigth_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
early = EarlyStopping(monitor='val_loss', mode='min', patience=20)
callbacks_list = [checkpoint]
callback = keras.callbacks.LearningRateScheduler(scheduler)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Training the model
print("[INFO] Training the model ...")
history = my_model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data= v_aug.flow(X_valid, y_valid), epochs=NUM_EPOCHS,
                  steps_per_epoch=len(X_train )// 16, callbacks=[early, callbacks_list, tensorboard_callback], verbose=1)

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
model_json = my_model.to_json()
with open("models/mvd_epi_Res.json", "w") as json_file:
    json_file.write(model_json)

K.clear_session()
