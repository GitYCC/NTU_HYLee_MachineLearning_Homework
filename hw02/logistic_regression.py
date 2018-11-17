#!python2.7
import math
import pickle

import numpy as np

from data_handling import get_train_set, get_test_set


class LogisticRegression(object):
    def __init__(self, input_dim, output_dim):
        self.__dim = (input_dim, output_dim)
        self.__W = np.zeros((1, input_dim+1, output_dim))
        self.__X = None
        self.__y = None

    def train(self, X, y,  # noqa: N803
              init_W=np.array([]), rate=0.01, alpha=0, epoch=1000, batch=None, validate_data=None):
        # print (X.reshape((4001, 57, 1)) * self.__W)
        self.__X = X
        self.__y = y
        num_data = X.shape[0]
        if not init_W.shape[0] == 0:
            if self.__W.shape == init_W.shape:
                self.__W = init_W
            else:
                raise ValueError('initial W has no correct dimension')

        if not batch:
            batch = num_data

        tot_batch = int(math.ceil(float(num_data) / float(batch)))

        for i in range(epoch):
            for j in range(tot_batch):
                batch_X = self.__X[j*batch:min(num_data, (j+1)*batch), :]
                batch_y = self.__y[j*batch:min(num_data, (j+1)*batch)]
                self.__W = self.update(batch_X, batch_y, self.__W, rate, alpha)

            msg = 'Epoch {:5d}: err = {:.6f} acc = {:.6f}'.format(
                i+1, self.err_insample(), self.accuarcy_insample())
            if validate_data:
                valid_X = validate_data[0]
                valid_y = validate_data[1]
                msg += ' , validate err = {:.6f} acc = {:.6f}'.format(
                    self.err(valid_X, valid_y), self.accuarcy(valid_X, valid_y))

            print msg

    def update(self, X, y, W, rate, alpha):  # noqa: N803
        # gradient = - y*(1-yi)*W
        X_argu = np.expand_dims(np.hstack((np.ones((X.shape[0], 1)), X)), axis=-1)
        z = np.sum(X_argu*W, axis=-2)
        y_pred = self.softmax(z)
        grad_tmp1 = (-1*y*(1-y_pred)).reshape((y.shape[0], 1, y.shape[1]))
        grad = np.sum(grad_tmp1*X_argu, axis=0)/y.shape[0] + alpha * W
        return W - rate * grad

    def predict(self, X):  # noqa: N803
        if X.shape[1] != self.__W.shape[1] - 1:
            raise ValueError('wrong dimension of X: should {}'.format(self.__W.shape[1]-1))
        X_argu = np.hstack((np.ones((X.shape[0], 1)), X))
        z = np.sum(np.expand_dims(X_argu, axis=-1)*self.__W, axis=-2)
        return self.softmax(z)

    def err_insample(self):
        if self.__X.size == 0 or self.__y.size == 0:
            raise RuntimeError('in-sample data not found')

        return self.err(self.__X, self.__y)

    def err(self, X, y):  # noqa: N803
        self._check_data(X, y)
        return self.cross_entropy(self.predict(X), y)

    def accuarcy_insample(self):
        if self.__X.size == 0 or self.__y.size == 0:
            raise RuntimeError('in-sample data not found')

        return self.accuarcy(self.__X, self.__y)

    def accuarcy(self, X, y):  # noqa: N803
        self._check_data(X, y)
        y_predict = self.predict(X)
        argmax_y = np.argmax(y, axis=1)
        argmax_y_predict = np.argmax(y_predict, axis=1)
        acc = float(np.sum(argmax_y == argmax_y_predict)) / float(y.shape[0])
        return acc

    @staticmethod
    def cross_entropy(ys, ys_hat):
        entropys = -1*np.sum(ys_hat*np.log(ys+1e-7), axis=1)
        return np.mean(entropys, axis=0)

    @staticmethod
    def softmax(zs):
        max_zs = np.expand_dims(np.max(zs, axis=-1), axis=-1)
        zs = zs - max_zs
        zs_tmp = np.exp(zs)
        return zs_tmp / np.expand_dims(np.sum(zs_tmp, axis=1), axis=-1)

    def _check_data(self, X, y):  # noqa: N803
        if X.shape[0] != y.shape[0]:
            raise ValueError('shape of X and y do not match')

    def save(self, path):
        with open(path,  'wb') as fw:
            pickle.dump(self,  fw,  pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path,  'rb') as fr:
            model = pickle.load(fr)
        return model

    @staticmethod
    def unit_test():
        lg = LogisticRegression(input_dim=3, output_dim=2)
        X_train = np.array([[0, 0, 0], [1, 1, 1]])
        y_train = np.array([[1, 0], [0, 1]])
        print lg.cross_entropy(ys=y_train, ys_hat=y_train)
        print lg.softmax(y_train)
        lg.train(X_train, y_train, rate=10, batch=1, epoch=5000)
        print lg.predict(X_train)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='HW2: Logistic Regression Training')
    parser.add_argument('--type',  metavar='TYPE',  type=str,  nargs='?',
                        help='type of job: \'train\' or \'test\'', required=True)
    parser.add_argument('--model',  metavar='MODEL',  type=str,  nargs='?',
                        help='path of output model', required=True)
    parser.add_argument('--output',  metavar='OUTPUT',  type=str,  nargs='?',
                        help='path of testing result', required=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.type == 'train':
        X_train, y_train, X_valid, y_valid = get_train_set()

        lg = LogisticRegression(input_dim=57, output_dim=2)
        lg.train(X_train, y_train, rate=7.7e-6, batch=10, epoch=20000, alpha=0,
                 validate_data=(X_valid, y_valid))
        lg.save(args.model)

    elif args.type == 'test':
        lg = LogisticRegression.load(args.model)
        X_test, ids = get_test_set()
        y_pred_label = np.argmax(lg.predict(X_test), axis=1)
        with open(args.output, 'w') as fw:
            fw.write('data_id,label\n')
            for i in range(ids.shape[0]):
                fw.write('{},{}\n'.format(ids[i], y_pred_label[i]))


if __name__ == '__main__':
    main()
