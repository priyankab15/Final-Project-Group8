from sklearn.preprocessing import OneHotEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from sklearn.preprocessing import LabelEncoder
from keras.layers import MaxPooling2D
from keras.models import Sequential
warnings.filterwarnings('ignore')
from keras.layers import Flatten
from keras.utils import np_utils
from keras.layers import Conv2D
from keras.layers import Dense
import pandas as pd
import numpy as np
import warnings
import cv2
import os


os.chdir("/home/ubuntu/ml2_project/Validation_images")
dir = ("/home/ubuntu/ml2_project/Validation_images")

p = 1
s = (34471, 60, 60, 3)
dim = (60,60)
allImage = np.zeros(s)
Class_order = []
labels = pd.read_csv("../new_labels1.csv")
labels = labels[["ImageID", "Class"]]
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
print(len(Class_order))
class_labels= np.array(Class_order)
imageData = (allImage/255)

encoder = LabelEncoder()
encoded_y= encoder.fit_transform(class_labels)
dummy_y = np_utils.to_categorical(encoded_y)
print dummy_y.shape

x_train, x_test, y_train, y_test = train_test_split(allImage, dummy_y, train_size=0.8, random_state=6)

print("X_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

num_classes = len(pd.unique(labels["Class"]))
Classes = pd.unique(labels["Class"])

width = 60
height = 60
num_channels = 3
flat = width * height
num_classes = 10


seed = 7
np.random.seed(seed)

model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(60, 60, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=80, epochs= 20)
score = model.evaluate(x_test, y_test, batch_size=80)

print("Test loss:",score[0])
print("Test accuracy", score[1])

