import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, classification_report
import pickle


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to input directory")
args = vars(ap.parse_args())

# Load info file
info_df = pd.read_csv('/Users/ebrahamalskaf/Documents/**ANATOMY_FUNCTION**/test_info.csv')

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
                    #gray = resize(gray, (224, 224))
                    #out = cv2.merge([gray, gray, gray])
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

(testX, testy) = load_images(args["directory"], info_df, im_size=224)
le = LabelEncoder().fit(testy)
testY = to_categorical(le.transform(testy), 3)

# Load trained model 1
json_file = open('models/fa_lenet_model1.json','r')
model1_json = json_file.read()
json_file.close()
model1 = model_from_json(model1_json)
model1.load_weights("models/#fa_lenet_predictor1_my_model.best.hdf5")

# Predict with model
preds1 = model1.predict(testX)
pred_test_cl1 = []
for p in preds1:
    pred = np.argmax(p, axis=0)
    pred_test_cl1.append(pred)
print(pred_test_cl1[:5])
#print(len(pred_test_cl))
#survival_yhat = list(info_df[args["target"]].values)
#print(len(survival_yhat))
#print(survival_yhat[:5])

prob_outputs1 = {
    "pred": pred_test_cl1,
    "actual_value": testy
}
prob_output_df1 = pd.DataFrame(prob_outputs1)
print(prob_output_df1.head())

# Evaluate model
print(classification_report(testY, pred_test_cl1))
print('LeNet reg ROCAUC score:',roc_auc_score(testY, pred_test_cl1))
print('LeNet reg Accuracy score:',accuracy_score(testY, pred_test_cl1))
print('LeNet reg score:',f1_score(testY, pred_test_cl1))

# Load trained model 2
json_file = open('models/fa_lenet_model2.json','r')
model2_json = json_file.read()
json_file.close()
model2 = model_from_json(model2_json)
model2.load_weights("models/#fa_lenet_predictor2_my_model.best.hdf5")

# Predict with model
preds2 = model2.predict(testX)
pred_test_cl2 = []
for p in preds2:
    pred = np.argmax(p, axis=0)
    pred_test_cl2.append(pred)
print(pred_test_cl2[:5])

prob_outputs2 = {
    "pred": pred_test_cl2,
    "actual_value": testY
}
prob_output_df2 = pd.DataFrame(prob_outputs2)
print(prob_output_df2.head())

# Evaluate model
print(classification_report(survival_yhat, pred_test_cl1))
print('LeNet ROCAUC score:',roc_auc_score(testY, pred_test_cl2))
print('LeNet Accuracy score:',accuracy_score(testY, pred_test_cl2))
print('LeNet F1 score:',f1_score(testY, pred_test_cl2))

# Load trained model 3
json_file = open('models/fa_lenetsgd_model.json','r')
model3_json = json_file.read()
json_file.close()
model3 = model_from_json(model3_json)
model3.load_weights("models/#fa_lenetsgd_predictor_my_model.best.hdf5")

# Predict with model
preds3 = model3.predict(testX)
pred_test_cl3 = []
for p in preds3:
    pred = np.argmax(p, axis=0)
    pred_test_cl3.append(pred)
print(pred_test_cl3[:5])

prob_outputs3 = {
    "pred": pred_test_cl3,
    "actual_value": testY
}
prob_output_df3 = pd.DataFrame(prob_outputs3)
print(prob_output_df3.head())

# Evaluate model
print(classification_report(testY, pred_test_cl3))
print('LeNet SGD ROCAUC score:',roc_auc_score(testY, pred_test_cl3))
print('LeNet SGD Accuracy score:',accuracy_score(testY, pred_test_cl3))
print('LeNet SGD F1 score:',f1_score(testY, pred_test_cl3))

# Load trained model 4
json_file = open('models/fa_lenetsgd100_model.json','r')
model4_json = json_file.read()
json_file.close()
model4 = model_from_json(model4_json)
model4.load_weights("models/#fa_lenetsgd100_predictor_my_model.best.hdf5")

# Predict with model
preds4 = model4.predict(testX)
pred_test_cl4 = []
for p in preds4:
    pred = np.argmax(p, axis=0)
    pred_test_cl4.append(pred)
print(pred_test_cl4[:5])

prob_outputs4 = {
    "pred": pred_test_cl4,
    "actual_value": testY
}
prob_output_df4 = pd.DataFrame(prob_outputs4)
print(prob_output_df4.head())

# Evaluate model
print(classification_report(testY, pred_test_cl4))
print('LeNet SGD long ROCAUC score:',roc_auc_score(testY, pred_test_cl4))
print('LeNet SGD long Accuracy score:',accuracy_score(testY, pred_test_cl4))
print('LeNet SGD long F1 score:',f1_score(testY, pred_test_cl4))

# Load trained model 5
json_file = open('models/fa_shallownet_model1.json','r')
model5_json = json_file.read()
json_file.close()
model5 = model_from_json(model5_json)
model5.load_weights("models/#fa_shallownet_predictor1_my_model.best.hdf5")

# Predict with model
preds5 = model5.predict(testX)
pred_test_cl5 = []
for p in preds5:
    pred = np.argmax(p, axis=0)
    pred_test_cl5.append(pred)
print(pred_test_cl5[:5])

prob_outputs5 = {
    "pred": pred_test_cl5,
    "actual_value": testY
}
prob_output_df5 = pd.DataFrame(prob_outputs4)
print(prob_output_df5.head())

# Evaluate model
print(classification_report(testY, pred_test_cl5))
print('ShallowNet ROCAUC score:',roc_auc_score(testY, pred_test_cl5))
print('ShallowNet Accuracy score:',accuracy_score(testY, pred_test_cl5))
print('ShallowNet F1 score:',f1_score(testY, pred_test_cl5))

# Plot RUC
fpr, tpr, _ = roc_curve(testY, preds1[:,1])
auc = round(roc_auc_score(testY, preds1[:,1]), 2)
plt.plot(fpr, tpr, label="LeNet with regularisation, AUC="+str(auc))
fpr, tpr, _ = roc_curve(testY, preds2[:,1])
auc = round(roc_auc_score(testY, preds2[:,1]), 2)
plt.plot(fpr, tpr, label="LeNet , AUC="+str(auc))
fpr, tpr, _ = roc_curve(testY, preds3[:,1])
auc = round(roc_auc_score(testY, preds3[:,1]), 2)
plt.plot(fpr, tpr, label="LeNet with SGD, AUC="+str(auc))
fpr, tpr, _ = roc_curve(testY, preds4[:,1])
auc = round(roc_auc_score(testY, preds4[:,1]), 2)
plt.plot(fpr, tpr, label="LeNet with SGO long training, AUC="+str(auc))
fpr, tpr, _ = roc_curve(testY, preds5[:,1])
auc = round(roc_auc_score(testY, preds5[:,1]), 2)
plt.plot(fpr, tpr, label="ShallowNet, AUC="+str(auc))
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Survival Models Comparison')
plt.show()
