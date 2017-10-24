#!python2.7
import sys,os

PATH = os.path.dirname(os.path.realpath(__file__))

from common import Tee
tee = Tee(os.path.join(PATH,'log_supervised_cnn.logg'), 'w')

import numpy as np 
import pandas as pd
import pickle
import math

from common import load_label
from common import split_data
from common import transform_channel
from common import data_augmentation

# preproc 
folder = os.path.join(PATH,'data')
X,Y = load_label(folder)
X = transform_channel(X,orig_mode="channels_first")

# split data
X_train,Y_train,X_valid,Y_valid = split_data(X,Y,ratio=0.9)


X_train,Y_train = data_augmentation(X_train,Y_train)





# small down dataset
#X, Y, X_out, Y_out = split_data(X,Y,ratio=0.5)

# standardize train data
#from sklearn.preprocessing import StandardScaler
#orig_shape = X_train.shape
#tmp_X_train = X_train.copy()
#tmp_X_train.shape = (orig_shape[0],orig_shape[1]*orig_shape[2]*orig_shape[3])
#scaler = StandardScaler().fit(tmp_X_train)
#tmp_X_train = scaler.transform(tmp_X_train)
#tmp_X_train.shape = orig_shape
#X_train = tmp_X_train

import keras
from models_supervised_cnn import SqueezeNet
from models_supervised_cnn import bryan_net
from models_supervised_cnn import YCNet
from models_supervised_cnn import YCNet2
from models_supervised_cnn import YCNet3

#model = SqueezeNet(10,inputs=(3,32,32))
#model = bryan_net(10,inputs=(32,32,3))
#model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(3,32,32))
#model = YCNet(10,inputs=(32,32,3))
model,batch_size = YCNet3(10,inputs=(32,32,3))
model.summary()


from keras.callbacks import EarlyStopping, ModelCheckpoint
from common import PlotLosses


model_dir =  os.path.join(os.path.dirname(os.path.realpath(__file__)),'model')
model_tmp_dir = os.path.join(model_dir,'tmp')
model_path = os.path.join(model_tmp_dir,'model_cnn_gen{epoch:02d}_loss{val_loss:.2f}.hdf5')
for file in os.listdir(model_tmp_dir):
    os.remove(os.path.join(model_tmp_dir,file))

import time
time_stamp = int(time.time()/1)

model.fit(X_train, Y_train,
          batch_size=batch_size, 
          epochs=40, 
          validation_data=(X_valid,Y_valid),
          callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True), 
                     EarlyStopping(monitor='val_loss', patience=3, mode='min'),
                     PlotLosses(output_img=os.path.join(model_dir,'loss_plot_{}.png'.format(time_stamp)))])

