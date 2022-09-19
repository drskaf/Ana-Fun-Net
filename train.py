import tensorflow as tf
import imutils
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from tensor_cnn import AlexNet, GoogLeNet, ResNet
from imutils import paths
from utils import get_concat_v_cut
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# Contruct argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of perfusion images")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# Initialise the list of data and labels
data = []
labels = []

# Loop over the input images and apply data augmentation
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 600), interpolation=cv2.INTER_CUBIC)
    image = img_to_array(image)
    aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.2, zoom_range=0.2, fill_mode="nearest")
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Normalise image pixels
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 10)

# Split dataset
trainX, testX, trainY, testY = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX,
	trainY, test_size=0.20)

# initialize the model
print("[INFO] compiling model...")
model = ResNet(INPUT_SHAPE=[600,200,1], OUTPUT=10)
opt = tf.keras.optimizers.SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(valX, valY),
	 steps_per_epoch=len(trainX)//32, epochs=50, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 50), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
