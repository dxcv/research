import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten


def build_model(nbr_features, loss='mean_squared_error', metrics='mean_squared_error', learning_rate=0.001):

    # Input Layer
    model = Sequential()  # 32, 16, 8, 4, 2
    model.add(Dense(32, kernel_initializer='normal', input_dim=nbr_features, activation='relu'))
    # hidden layers
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # output layer
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model

# model must be initialized with train_data.shape[1]
# model.summary()
# model.fit(training_dataset, training_dataset['target'], epochs=100)
# batch size default is 32
# test_loss, test_acc = model.evaluate(X_validate, y_validate)
#