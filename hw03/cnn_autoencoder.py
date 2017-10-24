#!python2.7
import sys,os

PATH = os.path.dirname(os.path.realpath(__file__))

from common import Tee
tee = Tee(os.path.join(PATH,'log_cnn_autoencoder.logg'), 'w')

import numpy as np 
import pandas as pd
import pickle
import math

from common import load_label, load_unlabel
from common import split_data
from common import transform_channel
from common import data_augmentation

from keras import backend as K


# label data preproc
folder = os.path.join(PATH,'data')
LX,LY = load_label(folder)
LX = transform_channel(LX,orig_mode="channels_first")
LX,LY,X_valid,Y_valid = split_data(LX,LY,ratio=0.9)


# unlabel data preproc
UX = load_unlabel(folder)
UX = transform_channel(UX,orig_mode="channels_first")

# load model
def cnn_autoencoder(nb_classes, inputs=(32,32,3), file_load_weights=None):
    import keras
    import h5py
    from keras.models import Model
    from keras.layers import Input, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Lambda

    def norm_relu(in_layer):
        return Activation('relu')(BatchNormalization(epsilon=1e-03)(in_layer))

    input_img = Input(shape=inputs) # 3,32x32
    
    norm0 = Lambda(lambda x: K.cast(x,dtype='float32')/255.0)(input_img)
    
    encoded = Conv2D(32,(3,3),padding='same')(norm0) # 32x32
    encoded = norm_relu(encoded)
    encoded = MaxPooling2D(pool_size=(2,2),strides=(2,2))(encoded) # 16x16

    encoded = Conv2D(16,(3,3),padding='same')(encoded) 
    encoded = norm_relu(encoded)


    encoded = Conv2D(4,(3,3),padding='same')(encoded) 
    encoded = norm_relu(encoded)
    code = MaxPooling2D(pool_size=(2,2),strides=(2,2))(encoded) # 8x8

    decoded = Conv2D(4,(3,3),padding='same')(code) 
    decoded = norm_relu(decoded)
    decoded = UpSampling2D(size=(2,2))(decoded) # 16x16
    
 

    decoded = Conv2D(16,(3,3),padding='same')(decoded) 
    decoded = norm_relu(decoded)

    decoded = Conv2D(32,(3,3),padding='same')(decoded) 
    decoded = norm_relu(decoded)
    decoded = UpSampling2D(size=(2,2))(decoded) # 32x32

    decoded = Conv2D(3,(1,1),padding='same',activation='sigmoid')(decoded) # 3,32x32
    
    
    ae = Model(inputs=input_img, outputs=decoded)
    #adam1 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ae.compile(loss='binary_crossentropy', optimizer='adam')

    ae_batch = 128
    
    classifier = Flatten()(code)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(1024, activation='relu')(classifier)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(256, activation='relu')(classifier)
    classifier = Dropout(0.5)(classifier)

    classifier = Dense(nb_classes, activation='softmax')(classifier)

    aednn = Model(inputs=input_img, outputs=classifier)
    adam2 = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    aednn.compile(loss='categorical_crossentropy', optimizer=adam2, metrics=['accuracy'])

    aednn_batch = 8

    return ae,aednn,ae_batch,aednn_batch


model_dir =  os.path.join(os.path.dirname(os.path.realpath(__file__)),'model')
model_tmp_dir = os.path.join(model_dir,'tmp')


 
ae, aednn, ae_batch, aednn_batch = cnn_autoencoder(10)
ae.summary()

PATH_AE = os.path.join(model_dir,"model_ae.hdf5")
if not os.path.exists(PATH_AE):
    train_ae_X    = np.concatenate((LX,UX),axis=0)
    train_ae_X, _ = data_augmentation(train_ae_X, np.ones((train_ae_X.shape[0],1)))

    normal_train_ae_X = np.asarray(train_ae_X,dtype='float32')/255.0
    normal_X_valid    = np.asarray(X_valid,dtype='float32')/255.0

    ae.fit( train_ae_X, normal_train_ae_X,
        batch_size=ae_batch,
        epochs=10, 
        validation_data=(X_valid, normal_X_valid), 
        verbose=1,
        )
    ae.save_weights(PATH_AE)
else:
    ae.load_weights(PATH_AE)


# freeze pretrain layers
for layer in ae.layers:
    layer.trainable = False


from keras.callbacks import EarlyStopping, ModelCheckpoint
from common import PlotLosses



model_path = os.path.join(model_tmp_dir,'model_aednn_gen{epoch:02d}_loss{val_loss:.2f}.hdf5')
for file in os.listdir(model_tmp_dir):
    os.remove(os.path.join(model_tmp_dir,file))


import time
time_stamp = int(time.time()/1)

aednn.summary()
train_aednn_X, train_aednn_Y = data_augmentation(LX, LY)

aednn.fit( train_aednn_X, train_aednn_Y,
        batch_size=aednn_batch,
        epochs=60, 
        validation_data=(X_valid,Y_valid), 
        verbose=1,
        callbacks=[#ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True), 
                   EarlyStopping(monitor='val_loss', patience=5, mode='min'),
                   PlotLosses(output_img=os.path.join(model_dir,'loss_plot_{}.png'.format(time_stamp)))])



