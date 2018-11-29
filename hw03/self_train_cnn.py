#!python2.7
import os
from shutil import copyfile
import tempfile

import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
import model_configs


def _load_model(model_config, path_restore):
    if model_config == 'vgg16':
        model = keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(3, 32, 32))
    else:
        func_get_custom_model = getattr(model_configs, model_config)
        model, batch_size = func_get_custom_model(10, inputs=(32, 32, 3))

    model.load_weights(path_restore)
    return model


def _select_unlabeled_above_relable(UX, model, relable_score):  # noqa: N803
    predicted = model.predict(UX, batch_size=64, verbose=1)
    relable_set = np.any(predicted > relable_score, axis=1)
    ux, uy = None, None
    if relable_set.shape[0] != 0:
        ux = UX[relable_set, :]
        uy = to_categorical(np.argmax(predicted[relable_set, :], axis=1), num_classes=10)
    return (ux, uy)


def _create_model(model_config, path_restore=None):
    if model_config == 'vgg16':
        model = keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(3, 32, 32))
        batch_size = 8
    else:
        func_get_custom_model = getattr(model_configs, model_config)
        model, batch_size = func_get_custom_model(10, inputs=(32, 32, 3))
    if path_restore is not None:
        model.load_weights(path_restore)
    return (model, batch_size)


def train(model_config, model_name):
    token = '{}_{}_{}'.format('st-cnn', model_config, model_name)

    checkpoint_dir = tempfile.mkdtemp(prefix=token+'_', dir=config.DIR_MODEL)
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

    # pretrain_model
    spv_token = '{}_{}_{}'.format('spv-cnn', model_config, model_name)
    pretrain_model_path = os.path.join(config.DIR_MODEL, 'MODEL_{}.hdf5'.format(spv_token))
    if os.path.exists(pretrain_model_path):
        model, batch_size = _create_model(model_config, path_restore=pretrain_model_path)
        model.summary()
    else:
        model, batch_size = _create_model(model_config)
        model.summary()
        model.fit(
            LX, LY,
            batch_size=batch_size,
            epochs=5,
            validation_data=(X_valid, Y_valid),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=3, mode='min'),
            ]
        )

    # ## self-training
    num_self_train = 20
    num_epochs = 10
    patience = 1

    path_best_checkpoint = None
    for st_round in range(1, 1+num_self_train):
        print('\n\n----- Round {} -----\n\n'.format(st_round))
        round_token = token + '_round{}'.format(st_round)
        path_loss_plot = os.path.join(checkpoint_dir, 'LOSS_{}.png'.format(round_token))
        checkpoint_path = os.path.join(
            checkpoint_dir,
            'check_round{}'.format(st_round) + '_gen{epoch:02d}_loss{val_loss:.2f}.hdf5'
        )

        # restore model
        if path_best_checkpoint is not None:
            model, batch_size = _create_model(model_config, path_restore=path_best_checkpoint)

        # add predicted unlabel data above relable_score
        relable_score_move = [0.975, 0.990]
        relable_score = round(
            relable_score_move[0] +
            (relable_score_move[1] - relable_score_move[0]) * st_round / num_self_train, 3)

        X_train, Y_train = LX, LY
        ux, uy = _select_unlabeled_above_relable(UX, model, relable_score)
        if ux is not None:
            X_train = np.concatenate((X_train, ux), axis=0)
            Y_train = np.concatenate((Y_train, uy), axis=0)
        X_train, Y_train = data_augmentation(X_train, Y_train)

        model.fit(
                X_train, Y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_data=(X_valid, Y_valid),
                callbacks=[
                    ModelCheckpoint(checkpoint_path, monitor='val_loss'),
                    EarlyStopping(monitor='val_loss', patience=patience, mode='min'),
                    PlotLosses(output_img=path_loss_plot)
                ]
        )
        del model

        # select best model
        checkpoints = filter(lambda x: '_loss' in x, os.listdir(checkpoint_dir))
        best_checkpoint = sorted(
            checkpoints,
            key=lambda x: float((x.split('_loss')[1]).replace('.hdf5', ''))
        )[0]
        path_best_checkpoint = os.path.join(checkpoint_dir, best_checkpoint)

    copyfile(path_best_checkpoint, model_path)


def evaluate(model_config, model_name):
    # load model
    token = '{}_{}_{}'.format('st-cnn', model_config, model_name)
    model_path = os.path.join(config.DIR_MODEL, 'MODEL_{}.hdf5'.format(token))
    model = _load_model(model_config, path_restore=model_path)

    # prepare testing data
    X_test = load_test(config.DIR_DATA)
    X_test = transform_channel(X_test, orig_mode='channels_first')
    y_test = to_categorical(load_test_ans(config.DIR_DATA))

    # evaluate
    loss_test, acc_test = model.evaluate(X_test, y_test, batch_size=32)
    print 'Test: loss={}, acc={} %'.format(loss_test, acc_test * 100.0)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='HW3: Self-train CNN')
    parser.add_argument('--type',  metavar='TYPE',  type=str,  nargs='?',
                        help='type of job: \'train\' or \'eval\'', required=True)
    parser.add_argument('--model_config',  metavar='MODEL',  type=str,  nargs='?',
                        help='model config', required=True)
    parser.add_argument('--model_name',  metavar='MODEL',  type=str,  nargs='?',
                        help='model name', required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.type == 'train':
        train(args.model_config, args.model_name)

    elif args.type == 'eval':
        evaluate(args.model_config, args.model_name)


if __name__ == '__main__':
    main()
