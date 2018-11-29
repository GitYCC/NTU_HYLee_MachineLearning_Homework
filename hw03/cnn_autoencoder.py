#!python2.7
import os
import tempfile

import numpy as np
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from keras.utils import to_categorical

import config
from common import (
    load_label,
    load_unlabel,
    split_data,
    transform_channel,
    data_augmentation,
    Tee,
    PlotLosses,
    load_test,
    load_test_ans,
)
import ae_classifier_configs


def _load_model(model_config, path_restore):
    func_get_custom_ae = getattr(ae_classifier_configs, model_config)
    ae = func_get_custom_ae(10, inputs=(32, 32, 3))
    model, _ = ae.get_ae_classifier()
    model.load_weights(path_restore)
    return model


def train(model_config, model_name):
    token = '{}_{}_{}'.format('ae-cnn', model_config, model_name)

    checkpoint_dir = tempfile.mkdtemp(prefix=token+'_', dir=config.DIR_MODEL)
    path_loss_plot = os.path.join(checkpoint_dir, 'LOSS_{}.png'.format(token))
    checkpoint_path = os.path.join(
        checkpoint_dir,
        'check_gen{epoch:02d}_loss{val_loss:.2f}.hdf5'
    )
    model_path = os.path.join(config.DIR_MODEL, 'MODEL_{}.hdf5'.format(token))
    tee = Tee(os.path.join(config.DIR_LOG, 'LOG_{}.logg'.format(token)), 'w')  # noqa: F841

    # ## preproc
    # label data preproc
    LX, LY = load_label(config.DIR_DATA)
    LX = transform_channel(LX, orig_mode='channels_first')
    LX, LY, X_valid, Y_valid = split_data(LX, LY, ratio=0.9)

    # unlabel data preproc
    UX = load_unlabel(config.DIR_DATA)
    UX = transform_channel(UX, orig_mode='channels_first')

    func_get_aec = getattr(ae_classifier_configs, model_config)
    autoencoder_classifier = func_get_aec(10, inputs=(32, 32, 3))

    # pretrain autoencoder
    train_ae_X = np.concatenate((LX, UX), axis=0)
    train_ae_X, _ = data_augmentation(train_ae_X, np.ones((train_ae_X.shape[0], 1)))

    normal_train_ae_X = np.asarray(train_ae_X, dtype='float32')/255.0
    normal_X_valid = np.asarray(X_valid, dtype='float32')/255.0

    ae, batch_ae = autoencoder_classifier.get_autoencoder()
    ae.summary()
    ae.fit(
        train_ae_X, normal_train_ae_X,
        batch_size=batch_ae,
        epochs=5,
        validation_data=(X_valid, normal_X_valid),
        verbose=1,
    )
    # autoencoder_classifier.freeze_ae_layers()

    # train

    train_X, train_Y = data_augmentation(LX, LY)

    ae_classifier, batch_ae_classifier = autoencoder_classifier.get_ae_classifier()
    ae_classifier.summary()
    ae_classifier.fit(
        train_X, train_Y,
        batch_size=batch_ae_classifier,
        epochs=60,
        validation_data=(X_valid, Y_valid),
        verbose=1,
        callbacks=[
            ModelCheckpoint(checkpoint_path, monitor='val_loss'),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=3, mode='min'),
            PlotLosses(output_img=path_loss_plot)
        ]
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='HW3: CNN Autoencoder')
    parser.add_argument('--type',  metavar='TYPE',  type=str,  nargs='?',
                        help='type of job: \'train\' or \'eval\'', required=True)
    parser.add_argument('--model_config',  metavar='MODEL',  type=str,  nargs='?',
                        help='model config', required=True)
    parser.add_argument('--model_name',  metavar='MODEL',  type=str,  nargs='?',
                        help='model name', required=True)

    args = parser.parse_args()
    return args


def evaluate(model_config, model_name):
    # load model
    token = '{}_{}_{}'.format('ae-cnn', model_config, model_name)
    model_path = os.path.join(config.DIR_MODEL, 'MODEL_{}.hdf5'.format(token))
    model = _load_model(model_config, path_restore=model_path)

    # prepare testing data
    X_test = load_test(config.DIR_DATA)
    X_test = transform_channel(X_test, orig_mode='channels_first')
    y_test = to_categorical(load_test_ans(config.DIR_DATA))

    # evaluate
    loss_test, acc_test = model.evaluate(X_test, y_test, batch_size=32)
    print 'Test: loss={}, acc={} %'.format(loss_test, acc_test * 100.0)


def main():
    args = parse_args()

    if args.type == 'train':
        train(args.model_config, args.model_name)

    elif args.type == 'eval':
        evaluate(args.model_config, args.model_name)


if __name__ == '__main__':
    main()
