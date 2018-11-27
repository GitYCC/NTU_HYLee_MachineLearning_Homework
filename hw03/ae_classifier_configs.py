from abc import ABCMeta, abstractmethod

import keras
from keras import backend as K  # noqa: N812
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    BatchNormalization,
    Flatten,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Dense,
    Dropout,
    Lambda,
)


class AEClassifierBase:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def _get_model_config(nb_classes, inputs=(32, 32, 3)):
        """Return (input_img, code, decoded, classifier)."""

    @staticmethod
    @abstractmethod
    def _compile_ae(ae):
        pass

    @staticmethod
    @abstractmethod
    def _compile_ae_classifier(ae_classifier):
        pass

    @property
    @abstractmethod
    def _ae_batch(self):
        pass

    @property
    @abstractmethod
    def _ae_classifier_batch(self):
        pass

    def __init__(self, nb_classes, inputs=(32, 32, 3)):
        self._nb_classes = nb_classes
        self._inputs = inputs

        input_img, code, decoded, classifier = self._get_model_config(
            nb_classes, inputs)
        self._input_img = input_img
        self._code = code
        self._decoded = decoded
        self._classifier = classifier
        self._ae = Model(inputs=self._input_img, outputs=self._decoded)
        self._compile_ae(self._ae)
        self._ae_classifier = Model(inputs=self._input_img, outputs=self._classifier)
        self._compile_ae_classifier(self._ae_classifier)

    def get_autoencoder(self):
        return self._ae, self._ae_batch

    def get_ae_classifier(self):
        return self._ae_classifier, self._ae_classifier_batch

    def freeze_ae_layers(self):
        if self._ae is not None:
            for layer in self._ae.layers:
                layer.trainable = False


class TestAEClassifier(AEClassifierBase):
    @staticmethod
    def _get_model_config(nb_classes, inputs=(32, 32, 3)):
        def norm_relu(in_layer):
            return Activation('relu')(BatchNormalization(epsilon=1e-03)(in_layer))

        input_img = Input(shape=inputs)  # 3, 32x32

        norm0 = Lambda(lambda x: K.cast(x, dtype='float32')/255.0)(input_img)

        encoded = Conv2D(8, (3, 3), padding='same')(norm0)  # 32x32
        encoded = norm_relu(encoded)
        encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoded)  # 16x16

        encoded = Conv2D(4, (3, 3), padding='same')(encoded)
        encoded = norm_relu(encoded)

        encoded = Conv2D(2, (3, 3), padding='same')(encoded)
        encoded = norm_relu(encoded)
        code = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoded)  # 8x8

        decoded = Conv2D(2, (3, 3), padding='same')(code)
        decoded = norm_relu(decoded)
        decoded = UpSampling2D(size=(2, 2))(decoded)  # 16x16

        decoded = Conv2D(4, (3, 3), padding='same')(decoded)
        decoded = norm_relu(decoded)

        decoded = Conv2D(8, (3, 3), padding='same')(decoded)
        decoded = norm_relu(decoded)
        decoded = UpSampling2D(size=(2, 2))(decoded)  # 32x32

        decoded = Conv2D(3, (1, 1), padding='same', activation='sigmoid')(decoded)  # 3, 32x32

        classifier = Flatten()(code)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(32, activation='relu')(classifier)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(16, activation='relu')(classifier)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(nb_classes, activation='softmax')(classifier)

        return input_img, code, decoded, classifier

    @staticmethod
    def _compile_ae(ae):
        adam = keras.optimizers.Adam(lr=0.0001)
        ae.compile(loss='mse', optimizer=adam)

    @staticmethod
    def _compile_ae_classifier(ae_classifier):
        adam = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ae_classifier.compile(
            loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    @property
    def _ae_batch(self):
        return 128

    @property
    def _ae_classifier_batch(self):
        return 8


class AutoencoderClassifier01(AEClassifierBase):
    @staticmethod
    def _get_model_config(nb_classes, inputs=(32, 32, 3)):
        def norm_relu(in_layer):
            return Activation('relu')(BatchNormalization(epsilon=1e-03)(in_layer))

        input_img = Input(shape=inputs)  # 3, 32x32

        norm0 = Lambda(lambda x: K.cast(x, dtype='float32')/255.0)(input_img)

        encoded = Conv2D(32, (3, 3), padding='same')(norm0)  # 32x32
        encoded = norm_relu(encoded)
        encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoded)  # 16x16

        encoded = Conv2D(16, (3, 3), padding='same')(encoded)
        encoded = norm_relu(encoded)

        encoded = Conv2D(4, (3, 3), padding='same')(encoded)
        encoded = norm_relu(encoded)
        code = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoded)  # 8x8

        decoded = Conv2D(4, (3, 3), padding='same')(code)
        decoded = norm_relu(decoded)
        decoded = UpSampling2D(size=(2, 2))(decoded)  # 16x16

        decoded = Conv2D(16, (3, 3), padding='same')(decoded)
        decoded = norm_relu(decoded)

        decoded = Conv2D(32, (3, 3), padding='same')(decoded)
        decoded = norm_relu(decoded)
        decoded = UpSampling2D(size=(2, 2))(decoded)  # 32x32

        decoded = Conv2D(3, (1, 1), padding='same', activation='sigmoid')(decoded)  # 3, 32x32

        classifier = Flatten()(code)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(256, activation='relu')(classifier)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(1024, activation='relu')(classifier)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(1024, activation='relu')(classifier)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(256, activation='relu')(classifier)
        classifier = Dropout(0.5)(classifier)
        classifier = Dense(nb_classes, activation='softmax')(classifier)

        return input_img, code, decoded, classifier

    @staticmethod
    def _compile_ae(ae):
        adam = keras.optimizers.Adam(lr=0.0001)
        ae.compile(loss='mse', optimizer=adam)

    @staticmethod
    def _compile_ae_classifier(ae_classifier):
        adam = keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ae_classifier.compile(
            loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    @property
    def _ae_batch(self):
        return 128

    @property
    def _ae_classifier_batch(self):
        return 8
