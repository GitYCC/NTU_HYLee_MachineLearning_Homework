#!python2.7
import os
import tempfile

import keras
# from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

import config
from common import (
    load_label,
    split_data,
    transform_channel,
    data_augmentation,
    Tee,
    PlotLosses,
)
import model_configs


def train(model_config, model_name):
    token = '{}_{}_{}'.format('spv-cnn', model_config, model_name)

    checkpoint_dir = tempfile.mkdtemp(prefix=token+'_', dir=config.DIR_MODEL)
    path_loss_plot = os.path.join(checkpoint_dir, 'LOSS_{}.png'.format(token))
    checkpoint_path = os.path.join(
        checkpoint_dir,
        'check_gen{epoch:02d}_loss{val_loss:.2f}.hdf5'
    )
    model_path = os.path.join(config.DIR_MODEL, 'MODEL_{}.hdf5'.format(token))
    tee = Tee(os.path.join(config.DIR_LOG, 'LOG_{}.logg'.format(token)), 'w')  # noqa: F841

    # ## preproc
    X, Y = load_label(config.DIR_DATA)
    X = transform_channel(X, orig_mode='channels_first')

    # ## split data
    X_train, Y_train, X_valid, Y_valid = split_data(X, Y, ratio=0.9)

    X_train, Y_train = data_augmentation(X_train, Y_train)

    # ## small down dataset
    # X, Y, X_out, Y_out = split_data(X, Y, ratio=0.5)

    # standardize train data

    # orig_shape = X_train.shape
    # tmp_X_train = X_train.copy()
    # tmp_X_train.shape = (orig_shape[0], orig_shape[1]*orig_shape[2]*orig_shape[3])
    # scaler = StandardScaler().fit(tmp_X_train)
    # tmp_X_train = scaler.transform(tmp_X_train)
    # tmp_X_train.shape = orig_shape
    # X_train = tmp_X_train

    if model_config == 'vgg16':
        model = keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet', input_tensor=None, input_shape=(3, 32, 32))
        batch_size = 8
    else:
        func_get_custom_model = getattr(model_configs, model_config)
        model, batch_size = func_get_custom_model(10, inputs=(32, 32, 3))

    model.summary()

    model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=3,
        validation_data=(X_valid, Y_valid),
        callbacks=[ModelCheckpoint(checkpoint_path, monitor='val_loss'),
                   ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min'),
                   EarlyStopping(monitor='val_loss', patience=3, mode='min'),
                   PlotLosses(output_img=path_loss_plot)]
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='HW3: Supervised CNN')
    parser.add_argument('--type',  metavar='TYPE',  type=str,  nargs='?',
                        help='type of job: \'train\' or \'eval\'', required=True)
    parser.add_argument('--model_config',  metavar='MODEL',  type=str,  nargs='?',
                        help='model config', required=True)
    parser.add_argument('--model_name',  metavar='MODEL',  type=str,  nargs='?',
                        help='model name', required=True)
    parser.add_argument('--output',  metavar='OUTPUT',  type=str,  nargs='?',
                        help='path of evaluation result', required=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.type == 'train':
        train(args.model_config, args.model_name)

    elif args.type == 'eval':
        pass


if __name__ == '__main__':
    main()
