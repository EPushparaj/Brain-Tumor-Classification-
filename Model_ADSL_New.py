import numpy as np
from keras import Input, Model
from keras.layers import Dense, Dropout, Concatenate, Flatten

from Evaluation_All import evaluation


# Define base models (you can define your own base models as needed)
def base_model_1(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def base_model_2(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Function to create the deep super learner network
def deep_super_learner(sol):
    input_shape = (244, 244, 1)  # Example input shape
    num_base_models = 4  # Number of base models in the super learner
    inputs = Input(shape=input_shape)
    base_outputs = []

    # Create base models and collect their outputs
    for i in range(num_base_models):
        base_model = base_model_1(input_shape) if i % 2 == 0 else base_model_2(input_shape)
        base_output = base_model(inputs)
        base_outputs.append(base_output)

    # Concatenate outputs of base models
    if len(base_outputs) > 1:
        merged = Concatenate()(base_outputs)
    else:
        merged = base_outputs[0]

    # Additional layers for the super learner
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(3, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def Model_ADSL(train_data, y_train, test_data, y_test, sol=None):
    print('ADSL')
    if sol is None:
        sol = [5, 5, 0]
    IMG_SIZE = 244
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

    model = deep_super_learner(sol[1])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, batch_size=sol[0], validation_split=0.1)

    pred = model.predict(X_test)
    Eval = evaluation(pred, y_test)
    return Eval