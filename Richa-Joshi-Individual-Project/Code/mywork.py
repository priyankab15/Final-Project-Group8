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