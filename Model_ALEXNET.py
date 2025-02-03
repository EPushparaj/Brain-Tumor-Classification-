import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from Evaluation_All import evaluation


def AlexNet():
    input_shape = (227, 227, 1)  # AlexNet takes input images of size 227x227x3
    num_classes = 3
    model = Sequential([
        # Layer 1
        Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer 2
        Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer 3
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        # Layer 4
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        # Layer 5
        Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Flatten the CNN output to feed it with fully connected layers
        Flatten(),
        # Layer 6
        Dense(4096, activation='relu'),
        Dropout(0.5),
        # Layer 7
        Dense(4096, activation='relu'),
        Dropout(0.5),
        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    return model


def Model_ALEXNET(train_data, y_train, test_data, y_test):
    print('Alexnet')
    IMG_SIZE = 227
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    X_train = Train_X.astype('float32') / 255
    X_test = Test_X.astype('float32') / 255
    # Instantiate the GoogleNet model
    model = AlexNet()

    # Print model summary
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=1, batch_size=10, validation_split=0.1)

    # Evaluate the model on test data
    pred = model.predict(X_test)
    Eval = evaluation(pred, y_test)
    return Eval
