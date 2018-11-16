#!python2.7
import os
import pickle
import math

import numpy as np
from keras.models import Sequential
from keras.layers import (
    Dense,
    Activation,
    Conv2D,
    MaxPooling2D,
    Flatten,
    # AveragePooling2D,
    Dropout,
    BatchNormalization,
)


def load_train_data(folder):
    with open(os.path.join(folder, 'all_label.p'), 'rb') as fr:
        label_data = pickle.load(fr)
    with open(os.path.join(folder, 'all_unlabel.p'), 'rb') as fr:
        unlabel_data = pickle.load(fr)

    label_data = np.array(label_data)
    unlabel_data = np.array(unlabel_data)
    return (label_data, unlabel_data)


folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
label_data, unlabel_data = load_train_data(folder)

X_train = label_data
X_train.shape = (10*500, 3072)
X_train.shape = (10*500, 32, 32, 3)

Y_train = np.array(
    [[1 if j == math.floor(i/500) else 0 for j in range(10)] for i in range(10*500)])

model = Sequential()
model.add(Conv2D(192, (5, 5), input_shape=(32, 32, 3)))
model.add(BatchNormalization(epsilon=1e-03))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (1, 1)))
model.add(BatchNormalization(epsilon=1e-03))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(384, (1, 1)))
model.add(BatchNormalization(epsilon=1e-03))
model.add(Activation('relu'))
model.add(Conv2D(256, (1, 1)))
model.add(BatchNormalization(epsilon=1e-03))
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3)))
model.add(BatchNormalization(epsilon=1e-03))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(512, kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=1e-03))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, batch_size=8, epochs=25,
          # callbacks=[ModelCheckpoint(modelPath, monitor='val_loss', save_best_only=True),
          # EarlyStopping(monitor='val_loss', patience=2, mode='min')]
          )
