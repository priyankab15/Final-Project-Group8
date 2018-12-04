import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import timedelta
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import warnings
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

os.chdir("/home/ubuntu/final_project_latest/Validation_images")
dir = ("/home/ubuntu/final_project_latest/Validation_images")
# num =0
p = 1
nuofimage=0
s = (34471, 60, 60, 3)
dim = (60,60)
allImage = np.zeros(s)
Class_order = []
labels = pd.read_csv("../new_labels1.csv")
labels = labels[["ImageID", "Class"]]
lab= ['Equipment', 'Automobile', 'Building', 'Person', 'Tree',
       'Electronics', 'Animal', 'Food_stuff', 'Clothes_accessories',
       'Furniture']
print("Labels:", lab)
print('Number of classes:', len(pd.unique(labels["Class"])))
for i in os.listdir(dir):
    name = os.path.basename(i)
    l = os.path.splitext(name)[0]
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if resized.shape == (60, 60, 3):
        a= np.array(resized)
        allImage[p-1] =a
        p = p+1
        x = labels.loc[labels["ImageID"] == l, "Class"]
        Class_order.append(x.values[0])



print("Shape of input:",allImage.shape)
# print(len(Class_order))
class_labels= np.array(Class_order)
print('Shape of labels:', class_labels.shape)
imageData = (allImage/255)

encoder = LabelEncoder()
encoded_y= encoder.fit_transform(class_labels)
dummy_y = np_utils.to_categorical(encoded_y)
print dummy_y.shape

# x_train, x_test, y_train, y_test = train_test_split(allImage, dummy_y, train_size=0.8, random_state=6)
#
# print("X_train shape:",x_train.shape)
# print("y_train shape:",y_train.shape)
# print("X_test shape:", x_test.shape)
# print("y_test shape:", y_test.shape)

num_classes = len(pd.unique(labels["Class"]))
Classes = pd.unique(labels["Class"])

width = 60
height = 60
num_channels = 3
flat = width * height
num_classes = 10

seed = 7
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(allImage, encoded_y):
  # create model

    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(60, 60, 3)))
    # model.add(Conv2D(32, (5,5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # sgd = Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.summary()
    history = model.fit(allImage[train], dummy_y[train], epochs=10, verbose=0, batch_size= 50,validation_data=(allImage[test], dummy_y[test]))
    model.fit(allImage[train], dummy_y[train], epochs=10, verbose=0, batch_size= 2)
    y_pred = model.predict(allImage[test])
    y_test_non_category = [ np.argmax(t) for t in dummy_y[test] ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]
    cm = confusion_matrix(y_test_non_category, y_predict_non_category)

    # model.fit(allImage[train], dummy_y[train], batch_size=50, epochs= 10)
    # history = model.fit(allImage, encoded_y, )
    # scores = model.evaluate(allImage[test], dummy_y[test], batch_size=32)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # cvscores.append(scores[1] * 100)

training_loss = history.history['loss']
test_loss = history.history['val_loss']
training_acc= history.history['acc']
test_acc = history.history['val_acc']
print(cm)
print(training_loss)
print(test_loss)
print(training_acc)
print(test_acc)
epochs = range(1, 11)
ax1 = plt.figure(1)
ax1.plt(epochs, training_acc, 'r--')
ax1.plt(epochs, test_acc, 'b--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

fig1 =plt.figure(2)
ax2 = fig1.add_subplot(111)
ax2.plt(epochs, training_loss, 'r--')
ax2.plt(epochs, test_loss, 'b--')
ax2.set_xlabel("Epoch")
plt.set_ylabel('Loss')
plt.show()

fig = plt.figure(3)
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for classifier')
fig.colorbar(cax)
ax.set_xticklabels(['']+ lab)
ax.set_yticklabels([''] + lab)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

