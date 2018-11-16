#!python2.7
import os
from shutil import copyfile
import time

import numpy as np

from common import (
    load_label,
    load_unlabel,
    split_data,
    transform_channel,
    data_augmentation,
    Tee,
)


PATH = os.path.dirname(os.path.realpath(__file__))
tee = Tee(os.path.join(PATH, 'log_self_train_cnn.logg'), 'w')

# label data preproc
folder = os.path.join(PATH, 'data')
LX, LY = load_label(folder)
LX = transform_channel(LX, orig_mode='channels_first')
LX, LY, X_valid, Y_valid = split_data(LX, LY, ratio=0.9)

# unlabel data preproc
UX = load_unlabel(folder)
UX = transform_channel(UX, orig_mode='channels_first')

# load model
from models_supervised_cnn import model_ycnet3

model_input = os.path.join(PATH, 'model', 'model_cnn_gen15_loss1.07_acc67.6.hdf5')  # path or None

if os.path.isfile(model_input):
    model, batch_size = model_ycnet3(10, inputs=(32, 32, 3), file_load_weights=model_input)
else:
    model, batch_size = model_ycnet3(10, inputs=(32, 32, 3))
    model.summary()


# ### self training CNN
from keras.callbacks import EarlyStopping, ModelCheckpoint
from common import PlotLosses
from keras.utils import to_categorical

# model store
model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
model_tmp_dir = os.path.join(model_dir, 'tmp')
model_path = os.path.join(model_tmp_dir, 'model_STCNN_gen{epoch:02d}_loss{val_loss:.2f}.hdf5')

for _file in os.listdir(model_tmp_dir):
    os.remove(os.path.join(model_tmp_dir, _file))

# training

num_self_train = 20

for i in range(num_self_train):

    print('\n\n----- Round {} -----\n\n'.format(i+1))

    X_train = LX
    Y_train = LY

    if i == 0:
        if model_input:
            continue
        num_epochs = 40
        patience = 5

    else:
        num_epochs = 10
        patience = 1

        # add predicted unlabel data above relable_score
        print('\n\nPredict Unlabel Data ...\n\n')
        uy = model.predict(UX, batch_size=64, verbose=1)

        relable_score_move = [0.975, 0.990]
        relable_score = round(
            relable_score_move[0] +
            (relable_score_move[1] - relable_score_move[0]) * i / num_self_train, 3)
        relable_set = np.any(uy > relable_score, axis=1)

        if relable_set.shape[0] != 0:
            ux = UX[relable_set, :]
            uy = to_categorical(np.argmax(uy[relable_set, :], axis=1), num_classes=10)
            print('\nAdd {} predicted unlabel data above {} relable\n'.format(
                uy.shape[0], relable_score))

            X_train = np.concatenate((X_train, ux), axis=0)
            Y_train = np.concatenate((Y_train, uy), axis=0)

    X_train, Y_train = data_augmentation(X_train, Y_train)

    time_stamp = int(time.time()/1)

    path_loss_plot = os.path.join(model_dir, 'loss_plot_{}.png'.format(time_stamp))
    model.fit(
              X_train, Y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(X_valid, Y_valid),
              callbacks=[
                  ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
                  EarlyStopping(monitor='val_loss', patience=patience, mode='min'),
                  PlotLosses(output_img=path_loss_plot)])
    del model

    # store model
    best_model = sorted(os.listdir(model_tmp_dir), reverse=True)[0]
    file_best_model = os.path.join(model_tmp_dir, best_model)
    new_file_best_model = os.path.join(model_dir, 'round{}_'.format(i)+best_model)
    copyfile(file_best_model, new_file_best_model)

    model, batch_size = model_ycnet3(10, inputs=(32, 32, 3), file_load_weights=new_file_best_model)

    for file in os.listdir(model_tmp_dir):
        os.remove(os.path.join(model_tmp_dir, file))
