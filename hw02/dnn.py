#!python2.7
import numpy as np
import keras

from data_handling import get_train_set, get_test_set


def get_dnn_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=10, input_dim=57))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(units=10))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(units=2))
    model.add(keras.layers.Activation('softmax'))

    # sgd = keras.optimizers.SGD(lr=7.7e-6, momentum=0.0, decay=0.0)
    adam = keras.optimizers.Adam(lr=0.00004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='HW2: DNN')
    parser.add_argument('--type', metavar='TYPE', type=str, nargs='?',
                        help='type of job: \'train\' or \'test\'', required=True)
    parser.add_argument('--model', metavar='MODEL', type=str, nargs='?',
                        help='path of output model', required=True)
    parser.add_argument('--output', metavar='OUTPUT', type=str, nargs='?',
                        help='path of testing result', required=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.type == 'train':
        X_train, y_train, X_valid, y_valid = get_train_set()
        model = get_dnn_model()
        model.fit(X_train, y_train, batch_size=10, epochs=300, validation_data=(X_valid, y_valid))
        model.save_weights(args.model)

    elif args.type == 'test':
        model = get_dnn_model()
        model.load_weights(args.model)
        X_test, ids = get_test_set()
        y_pred_label = np.argmax(model.predict(X_test, batch_size=64, verbose=1), axis=1)
        with open(args.output, 'w') as fw:
            fw.write('data_id,label\n')
            for i in range(ids.shape[0]):
                fw.write('{},{}\n'.format(ids[i], y_pred_label[i]))


if __name__ == '__main__':
    main()
