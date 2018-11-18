import keras
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    Activation,
    BatchNormalization,
    AveragePooling2D,
    Flatten,
    Dropout,
    Concatenate,
    Convolution2D,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    Dense
)
# from keras.layers.advanced_activations import LeakyReLU


def test(nb_classes, inputs=(32, 32, 3), file_load_weights=None):
    def norm_relu(in_layer):
        return Activation('relu')(BatchNormalization(epsilon=1e-03)(in_layer))

    input_img = Input(shape=inputs)
    norm0 = BatchNormalization(epsilon=1e-03)(input_img)

    conv1_1 = Conv2D(8, (5, 5), padding='same')(norm0)
    norm_relu1_1 = norm_relu(conv1_1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(norm_relu1_1)
    dropout1 = Dropout(0.5)(maxpool1)

    conv2_1 = Conv2D(4, (5, 5), padding='same')(dropout1)
    norm_relu2_1 = norm_relu(conv2_1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(norm_relu2_1)
    dropout2 = Dropout(0.5)(maxpool2)

    conv3_1 = Conv2D(10, (3, 3), padding='same')(dropout2)
    norm_relu3_1 = norm_relu(conv3_1)
    avepool3 = AveragePooling2D(pool_size=(8, 8), strides=(1, 1))(norm_relu3_1)

    flatten4 = Flatten()(avepool3)
    softmax4 = Activation('softmax')(flatten4)

    model = Model(inputs=input_img, outputs=softmax4)

    if file_load_weights:
        model.load_weights(file_load_weights)

    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    batch_size = 16

    return (model, batch_size)


def ycnet3(nb_classes, inputs=(32, 32, 3), file_load_weights=None):
    def norm_relu(in_layer):
        return Activation('relu')(BatchNormalization(epsilon=1e-03)(in_layer))

    input_img = Input(shape=inputs)
    norm0 = BatchNormalization(epsilon=1e-03)(input_img)

    conv1_1 = Conv2D(192, (5, 5), padding='same')(norm0)
    norm_relu1_1 = norm_relu(conv1_1)
    conv1_2 = Conv2D(160, (1, 1), padding='same')(norm_relu1_1)
    norm_relu1_2 = norm_relu(conv1_2)
    conv1_3 = Conv2D(96, (1, 1), padding='same')(norm_relu1_2)
    norm_relu1_3 = norm_relu(conv1_3)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(norm_relu1_3)
    dropout1 = Dropout(0.5)(maxpool1)

    conv2_1 = Conv2D(192, (5, 5), padding='same')(dropout1)
    norm_relu2_1 = norm_relu(conv2_1)
    conv2_2 = Conv2D(192, (1, 1), padding='same')(norm_relu2_1)
    norm_relu2_2 = norm_relu(conv2_2)
    conv2_3 = Conv2D(192, (1, 1), padding='same')(norm_relu2_2)
    norm_relu2_3 = norm_relu(conv2_3)
    padding2_2 = ZeroPadding2D(padding=(1, 1))(norm_relu2_3)
    avepool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(padding2_2)
    dropout2 = Dropout(0.5)(avepool2)

    conv3_1 = Conv2D(192, (3, 3), padding='same')(dropout2)
    norm_relu3_1 = norm_relu(conv3_1)
    conv3_2 = Conv2D(192, (1, 1), padding='same')(norm_relu3_1)
    norm_relu3_2 = norm_relu(conv3_2)
    conv3_3 = Conv2D(10, (1, 1), padding='same')(norm_relu3_2)
    norm_relu3_3 = norm_relu(conv3_3)
    avepool3 = AveragePooling2D(pool_size=(8, 8), strides=(1, 1))(norm_relu3_3)

    flatten4 = Flatten()(avepool3)
    softmax4 = Activation('softmax')(flatten4)

    model = Model(inputs=input_img, outputs=softmax4)

    if file_load_weights:
        model.load_weights(file_load_weights)

    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    batch_size = 16

    return (model, batch_size)


def ycnet2(nb_classes, inputs=(32, 32, 3)):
    input_img = Input(shape=inputs)

    norm0 = BatchNormalization(epsilon=1e-03)(input_img)

    conv1_1 = Conv2D(64, (5, 5), activation='relu', name='conv1_1')(norm0)
    norm1 = BatchNormalization(epsilon=1e-03)(conv1_1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), name='maxpool1')(norm1)

    conv2_1 = Conv2D(64, (2, 2), activation='relu', name='conv2_1')(maxpool1)
    conv2_2 = Conv2D(64, (2, 2), activation='relu', name='conv2_4')(conv2_1)
    norm2 = BatchNormalization(epsilon=1e-03)(conv2_2)

    conv3_0 = Conv2D(128, (1, 1), strides=(2, 2), activation='relu', name='conv3_0')(norm2)
    conv3_1 = Conv2D(128, (1, 1), activation='relu', name='conv3_1')(conv3_0)
    conv3_2 = Conv2D(128, (1, 1), activation='relu', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(128, (1, 1), activation='relu', name='conv3_3')(conv3_2)
    conv3_4 = Conv2D(128, (1, 1), activation='relu', name='conv3_4')(conv3_3)
    norm3 = BatchNormalization(epsilon=1e-03)(conv3_4)

    conv4_0 = Conv2D(256, (1, 1), strides=(2, 2), activation='relu', name='conv4_0')(norm3)
    conv4_1 = Conv2D(256, (1, 1), activation='relu', name='conv4_1')(conv4_0)
    conv4_2 = Conv2D(256, (1, 1), activation='relu', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(256, (1, 1), activation='relu', name='conv4_3')(conv4_2)
    conv4_4 = Conv2D(256, (1, 1), activation='relu', name='conv4_4')(conv4_3)

    # The size should match the output of conv10
    avgpool4 = AveragePooling2D((3, 3), name='avgpool4')(conv4_4)

    flatten4 = Flatten(name='flatten')(avgpool4)

    softmax = Dense(nb_classes, activation='softmax', name='predictions')(flatten4)

    model = Model(inputs=input_img, outputs=softmax)

    # adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 8

    return (model, batch_size)


def ycnet(nb_classes, inputs=(32, 32, 3)):
    def conv2d_norm_relu(n_filter, size_filter, inputs):
        w_filter, h_filter = size_filter
        layers = inputs
        layers = Conv2D(n_filter, (w_filter, h_filter))(layers)
        layers = BatchNormalization(epsilon=1e-03)(layers)
        layers = Activation(activation='relu')(layers)
        return layers

    input_img = Input(shape=inputs)

    conv_norm_relu1 = conv2d_norm_relu(192, (5, 5), input_img)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_norm_relu1)

    conv_norm_relu2 = conv2d_norm_relu(256, (1, 1), maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_norm_relu2)

    conv_norm_relu3_1 = conv2d_norm_relu(384, (1, 1), maxpool2)
    conv_norm_relu3_2 = conv2d_norm_relu(256, (1, 1), conv_norm_relu3_1)
    conv_norm_relu3_3 = conv2d_norm_relu(512, (3, 3), conv_norm_relu3_2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_norm_relu3_3)

    flatten4 = Flatten()(maxpool3)

    fc5 = Dense(512, activation='relu', kernel_initializer='he_normal')(flatten4)
    dropout5 = Dropout(0.5)(fc5)

    softmax = Dense(nb_classes, activation='softmax')(dropout5)

    model = Model(inputs=input_img, outputs=softmax)

    # adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 8

    return (model, batch_size)


def bryannet(nb_classes, inputs=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(192, (5, 5), input_shape=inputs))
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
    model.add(Dense(nb_classes, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 8

    return (model, batch_size)


def squeeze_net(nb_classes, inputs=(3, 224, 224)):
    """Use Keras to implement squeeze net(arXiv 1602.07360).

    Args:
        nb_classes: total number of final categories
        inputs: shape of the input images (channel, cols, rows)

    """
    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format='channels_first')(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format='channels_first')(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format='channels_first')(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format='channels_first')(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format='channels_first')(fire2_squeeze)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format='channels_first')(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format='channels_first')(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format='channels_first')(fire3_squeeze)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

#    fire4_squeeze = Convolution2D(
#        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire4_squeeze',
#        data_format='channels_first')(merge3)
#    fire4_expand1 = Convolution2D(
#        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire4_expand1',
#        data_format='channels_first')(fire4_squeeze)
#    fire4_expand2 = Convolution2D(
#        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire4_expand2',
#        data_format='channels_first')(fire4_squeeze)
#    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
#    maxpool4 = MaxPooling2D(
#        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
#        data_format='channels_first')(merge4)

#    fire5_squeeze = Convolution2D(
#        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire5_squeeze',
#        data_format='channels_first')(maxpool4)
#    fire5_expand1 = Convolution2D(
#        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire5_expand1',
#        data_format='channels_first')(fire5_squeeze)
#    fire5_expand2 = Convolution2D(
#        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire5_expand2',
#        data_format='channels_first')(fire5_squeeze)
#    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])
#
#    fire6_squeeze = Convolution2D(
#        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire6_squeeze',
#        data_format='channels_first')(merge5)
#    fire6_expand1 = Convolution2D(
#        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire6_expand1',
#        data_format='channels_first')(fire6_squeeze)
#    fire6_expand2 = Convolution2D(
#        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire6_expand2',
#        data_format='channels_first')(fire6_squeeze)
#    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])
#
#    fire7_squeeze = Convolution2D(
#        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire7_squeeze',
#        data_format='channels_first')(merge6)
#    fire7_expand1 = Convolution2D(
#        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire7_expand1',
#        data_format='channels_first')(fire7_squeeze)
#    fire7_expand2 = Convolution2D(
#        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire7_expand2',
#        data_format='channels_first')(fire7_squeeze)
#    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

#    fire8_squeeze = Convolution2D(
#        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire8_squeeze',
#        data_format='channels_first')(merge7)
#    fire8_expand1 = Convolution2D(
#        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire8_expand1',
#        data_format='channels_first')(fire8_squeeze)
#    fire8_expand2 = Convolution2D(
#        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire8_expand2',
#        data_format='channels_first')(fire8_squeeze)
#    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

#    maxpool8 = MaxPooling2D(
#        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
#        data_format='channels_first')(merge8)
#    fire9_squeeze = Convolution2D(
#        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire9_squeeze',
#        data_format='channels_first')(maxpool8)
#    fire9_expand1 = Convolution2D(
#        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire9_expand1',
#        data_format='channels_first')(fire9_squeeze)
#    fire9_expand2 = Convolution2D(
#        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
#        padding='same', name='fire9_expand2',
#        data_format='channels_first')(fire9_squeeze)
#    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge3)
    conv10 = Convolution2D(
        nb_classes, (1, 1), kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format='channels_first')(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D(
        (7, 7), name='avgpool10',
        data_format='channels_first')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation('softmax', name='softmax')(flatten)

    model = Model(inputs=input_img, outputs=softmax)

    adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    batch_size = 8

    return (model, batch_size)
