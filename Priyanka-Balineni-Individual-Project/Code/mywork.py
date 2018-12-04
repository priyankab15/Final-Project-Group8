# Tensorflow

s = (34471, width, height, 3)
allImage = np.zeros(s)
image_id = []
dim = (width, height)
Class_order = []
labels = pd.read_csv("../new_labels1.csv")
labels_raw = labels[["ImageID", "Class"]]
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
        val= x.values
        Class_order.append(val[0])

x = tf.placeholder(tf.float32, shape=[None, 60, 60, 3], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

# is_training = tf.placeholder(dtype=tf.bool, name='is_training')

x_image = tf.reshape(x, [-1, 60, 60, 3])  # -1 put everything as 1 array

y_true_cls = tf.argmax(y_true, axis=1)

keep_prob_fc=tf.placeholder(tf.float32)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

 batch_size = 50
    no_of_batches = int(x_train.shape[0]/batch_size)

    for epoch in range(10):
        total_iterations += 1
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
        # if (epoch % 1 ==0):
        #     print('Epoch:{}, Training Accuracy: {}, Testing Accuracy:{}'.format(epoch, acc_train, acc_test))
t1 = time.clock()
print("Time taken to run the code:", t1-t0)
print('Epoch:{}, Training Accuracy: {}, Testing Accuracy:{}'.format(epoch, acc_train, acc_test))
print(test_loss[1])
plt.plot(test_loss, label = "Testing  loss")
plt.plot(train_loss, label ="Training loss")
plt.legend()
plt.show()

# Keras
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(allImage, encoded_y):

    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(60, 60, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    history = model.fit(allImage[train], dummy_y[train], epochs=10, verbose=0, batch_size= 50,validation_data=(allImage[test], dummy_y[test]))
    model.fit(allImage[train], dummy_y[train], epochs=10, verbose=0, batch_size= 2)
    y_pred = model.predict(allImage[test])
    y_test_non_category = [ np.argmax(t) for t in dummy_y[test] ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred ]
    cm = confusion_matrix(y_test_non_category, y_predict_non_category)


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