#!python2.7
import numpy as np
import pandas as pd

data = '~/Documents/DeepLearning/NTU_HYLee_MachineLearning/Homework/hw02/spam_data/spam_train.csv'

COLUMNS = [
        'data_id', 'Feature_make', 'Feature_address', 'Feature_all', 'Feature_3d',
        'Feature_our', 'Feature_over', 'Feature_remove', 'Feature_internet', 'Feature_order',
        'Feature_mail', 'Feature_receive', 'Feature_will', 'Feature_people', 'Feature_report',
        'Feature_addresses', 'Feature_free', 'Feature_business', 'Feature_email', 'Feature_you',
        'Feature_credit', 'Feature_your', 'Feature_font', 'Feature_000', 'Feature_money',
        'Feature_hp', 'Feature_hpl', 'Feature_george', 'Feature_650', 'Feature_lab',
        'Feature_labs', 'Feature_telnet', 'Feature_857', 'Feature_data', 'Feature_415',
        'Feature_85', 'Feature_echnology', 'Feature_1999', 'Feature_parts', 'Feature_pm',
        'Feature_direct', 'Feature_cs', 'Feature_meeting', 'Feature_original', 'Feature_project',
        'Feature_re', 'Feature_edu', 'Feature_table', 'Feature_conference', 'Feature_;',
        'Feature_(', 'Feature_[', 'Feature_!', 'Feature_$', 'Feature_#',
        'Feature_capital_run_length_average', 'Feature_capital_run_length_longest',
        'Feature_capital_run_length_total', 'label'
        ]

df = pd.read_csv(data, names=COLUMNS)
X = np.array(df.drop(['data_id', 'label'], axis=1))
y = np.hstack((np.array(df[['label']]), 1-np.array(df[['label']])))

ratio = 0.8
num_data = X.shape[0]
num_train = int(ratio * num_data)
num_valid = num_data - num_train

X_train = X[0:num_train, :]
y_train = y[0:num_train, :]

X_valid = X[num_train:, :]
y_valid = y[num_train:, :]


import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=10, input_dim=57))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(units=10))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(units=2))
model.add(keras.layers.Activation('softmax'))

# sgd = keras.optimizers.SGD(lr=7.7e-6,  momentum=0.0,  decay=0.0)
adam = keras.optimizers.Adam(lr=0.00004,  beta_1=0.9,  beta_2=0.999,  epsilon=1e-08,  decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=300, validation_data=(X_valid, y_valid))
