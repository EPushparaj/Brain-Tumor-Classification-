from keras import Input
from keras.layers import Conv2D, BatchNormalization, Dense, \
    MaxPooling2D, Flatten, Activation, Concatenate
import numpy as np
from Evaluation_All import evaluation



import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Concatenate

def hierarchical_decoupling_conv(inputs, filters, kernel_size, alpha=0.5):
    local_filters = int(filters * alpha)
    global_filters = filters - local_filters

    # Local branch
    local_features = Conv2D(local_filters, kernel_size, padding='same')(inputs)

    # Global branch
    global_features = Conv2D(global_filters, kernel_size, padding='same')(inputs)

    # Concatenate local and global features
    combined_features = Concatenate()([local_features, global_features])

    # Batch normalization and ReLU activation
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)

    return combined_features

def HDCN():
    input_shape = (224, 224, 1)
    inputs = Input(shape=input_shape)

    # First HDC block
    x1 = hierarchical_decoupling_conv(inputs, filters=64, kernel_size=(3, 3), alpha=0.5)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    # Second HDC block
    x2 = hierarchical_decoupling_conv(x1, filters=128, kernel_size=(3, 3), alpha=0.5)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)

    # Flatten before fully connected layers
    x = Flatten()(x2)

    # Fully connected layers
    x = Dense(3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='HDCN')
    return model

def Model_HDCNet(train_data, y_train, test_data, y_test):
    print('HDCnet')
    IMG_SIZE = 224
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
    # Instantiate the HDCN model
    model = HDCN()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=1, batch_size=10, validation_split=0.1)

    pred = model.predict(X_test)
    Eval = evaluation(pred, y_test)
    return Eval