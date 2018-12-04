import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import cv2
import time
import os

warnings.filterwarnings('ignore')

# Defining the directory
os.chdir("/home/ubuntu/ml2_project/Validation_images")
dir = ("/home/ubuntu/ml2_project/Validation_images")

# Resizing the images
p = 1
#numofima= 0
width = int(60)
height = int(60)
Images_info = []
s = (34471, width, height, 3)
allImage = np.zeros(s)
image_id = []
dim = (width, height)
Class_order = []
labels = pd.read_csv("../new_labels1.csv")
labels_raw = labels[["ImageID", "Class"]]
imageid = labels["ImageID"].values
print('Number of classes:', len(pd.unique(labels_raw["Class"])))
number_images_shape= 0
for i in os.listdir(dir):
    name = os.path.basename(i)
    l = os.path.splitext(name)[0]
    image_id.append(i)
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if resized.shape == (60, 60, 3):
        number_images_shape= number_images_shape + 1
        a= np.array(resized)
        # clip= np.clip(a/255.0, 0.0, 1.0)
        allImage[p-1] =a
        p = p+1
        x = labels.loc[labels["ImageID"] == l, "Class"]
        # print x
        val= x.values
        # print val
        Class_order.append(x.values[0])

print("Number of images with shape (60,60,3):",number_images_shape)
print("Images shape:",allImage.shape)
print(len(Class_order))
# print(Class_order)
class_labels= np.array(Class_order)
print("Class labels shape:",class_labels.shape)
imageData = (allImage/255)

# One hot encoding
encoder = LabelEncoder()
encoder.fit(class_labels)
encoded_y= encoder.transform(class_labels)
dummy_y = np_utils.to_categorical(encoded_y)
print (dummy_y.shape)

# Split the dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(allImage, dummy_y, train_size=0.8, random_state=6)

# Printing the shape
print("X_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Unique labels
num_classes = len(pd.unique(labels["Class"]))
Classes = pd.unique(labels["Class"])
# print("Classes",Classes)

# Defining the size of image
width = 60
height = 60
num_channels = 3
flat = width * height
num_classes = 10

x = tf.placeholder(tf.float32, shape=[None, 60, 60, 3], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

x_image = tf.reshape(x, [-1, 60, 60, 3])  # -1 put everything as 1 array

y_true_cls = tf.argmax(y_true, axis=1)

keep_prob_fc= tf.placeholder(tf.float32)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, output_fm,  use_pooling=True):

    shape = [filter_size, filter_size, num_input_channels, output_fm]
    weights = new_weights(shape=shape)
    biases = new_biases(length=output_fm)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases
    layer = tf.nn.relu(layer)
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, use_dropout=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    if use_dropout:
        layer = tf.nn.dropout(layer, keep_prob_fc)

    return layer

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5.
num_filters1 = 32         # Using 32 filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 64 of these filters.

# Convolutional Layer 3.
filter_size3 = 3          # Convolution filters are 3 x 3 pixels.
num_filters3 = 128        # There are 128 of these filters.

filter_size4 = 3          # Convolution filters are 3 x 3 pixels.
num_filters4 = 256        # There are  256 of these filters.

# Fully-connected layer.
fc_size = 500

# Define the layers
# with tf.device('/gpu:0'):
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=3,
                   filter_size=filter_size1,
                   output_fm=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   output_fm=num_filters2,
                   use_pooling=True)

layer_conv3, weights_conv3 =\
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   output_fm=num_filters3,
                   use_pooling=True)

layer_conv4, weights_conv4 = \
    new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size4,
                   output_fm=num_filters4,
                   use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv4)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True,
                         use_dropout=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False,
                         use_dropout=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
y_true_cls= tf.argmax(y_true, axis=1)

# Transfer function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Time function to record a time to run the model
t0 = time.clock()

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

    train_loss=[]
    test_loss =[]

    batch_size = 50
    no_of_batches = int(x_train.shape[0]/batch_size)

    for epoch in range(10):
        ptr= 0
        for i in range(no_of_batches):
            x_batch, y_batch = x_train[ptr: ptr+batch_size], y_train[ptr: ptr+batch_size]
            ptr += batch_size

            sess.run(train, feed_dict={x: x_batch, y_true: y_batch, keep_prob_fc: 1.0})

        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(accuracy)
        acc_train = sess.run(accuracy, {x: x_train, y_true:y_train, keep_prob_fc: 1.0})
        acc_test = sess.run(accuracy, {x: x_test, y_true:y_test, keep_prob_fc: 1.0})
        train_loss.append(acc_train)
        test_loss.append(acc_test)
        # writer = tf.summary.FileWriter('./my_graph_2', sess.graph)
        # writer.close()
        # sess.close()
        if (epoch % 1 == 0):
            print('Epoch:{}, Training Accuracy: {}, Testing Accuracy:{}'.format(epoch, acc_train, acc_test))

t1 = time.clock()
print train_loss
print test_loss
print("Time taken to run the code:", t1-t0)
print('Epoch:{}, Training Accuracy: {}, Testing Accuracy:{}'.format(epoch, train_loss, test_loss))
print(test_loss[9])
plt.plot(test_loss, label = "Testing loss")
plt.plot(train_loss, label ="Training loss")
plt.legend()
plt.show()




