import numpy as np
from Evaluation_All import evaluation
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, concatenate


def inception_module(x, filters):
    # 1x1 convolution
    conv1x1 = Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same', activation='relu')(x)

    # 3x3 convolution
    conv3x3 = Conv2D(filters=filters[1], kernel_size=(1, 1), padding='same', activation='relu')(x)
    conv3x3 = Conv2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu')(conv3x3)

    # 5x5 convolution
    conv5x5 = Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same', activation='relu')(x)
    conv5x5 = Conv2D(filters=filters[4], kernel_size=(5, 5), padding='same', activation='relu')(conv5x5)

    # Max pooling
    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    pool = Conv2D(filters=filters[5], kernel_size=(1, 1), padding='same', activation='relu')(pool)

    # Concatenate the outputs (inception)
    output = concatenate([conv1x1, conv3x3, conv5x5, pool], axis=-1)
    return output


def googlenet():
    input_layer = Input(shape=(224, 224, 1))

    # Initial conv layer
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Two inception modules
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])

    # Max pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Two more inception modules
    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = inception_module(x, [112, 144, 288, 32, 64, 64])
    x = inception_module(x, [256, 160, 320, 32, 128, 128])

    # Max pooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Final inception module
    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = inception_module(x, [384, 192, 384, 48, 128, 128])

    # Average pooling
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x)
    x = Dropout(0.4)(x)

    # Flatten and dense layer for classification
    x = Flatten()(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=outputs)
    return model


def Model_GoogleNet(train_data, y_train, test_data, y_test):
    print("GoogleNet")
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
    # Instantiate the GoogleNet model
    model = googlenet()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=1, batch_size=10, validation_split=0.1)
    pred = model.predict(X_test)
    Eval = evaluation(pred, y_test)
    return Eval
