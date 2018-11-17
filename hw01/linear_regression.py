#!/usr/local/bin/python2.7
import math

import numpy as np


class LinearRegression(object):
    def __init__(self):
        self.__wt = None
        self.__b = None
        self.__X = None
        self.__y = None

    @property
    def wt(self):
        return self.__wt

    @property
    def b(self):
        return self.__b

    def train_by_pseudo_inverse(self, X, y, alpha=0, validate_data=None):  # noqa: N803
        self._check_data(X, y)

        self.__X = X
        self.__y = y

        num_data = X.shape[0]
        dim_wt = X.shape[1]

        argu_X = np.hstack((np.ones(num_data).reshape(num_data, 1), X))

        # calculate pseudo-inverse
        A_plus = np.dot(np.linalg.inv(np.dot(argu_X.T, argu_X)+alpha*np.eye(dim_wt+1)), argu_X.T)

        wt_b = np.dot(A_plus, y)
        self.__b = wt_b[0]
        self.__wt = wt_b[1:]

        if validate_data:
            print 'Pseudo-Inverse: err = {:.6f} validate = {:.6f}'.format(
                self.err_insample(),
                self.err(validate_data[0], validate_data[1]))
        else:
            print 'Pseudo-Inverse: err = {:.6f}'.format(self.err_insample())

    def train_by_gradient_descent(self, X, y, init_wt=np.array([]), init_b=0,  # noqa: N803
              rate=0.01, alpha=0, epoch=1000, batch=None, validate_data=None):
        self._check_data(X, y)

        if init_wt.size == 0:
            init_wt = np.zeros(X.shape[1])

        self.__X = X
        self.__y = y

        self.__wt = init_wt
        self.__b = init_b

        num_data = X.shape[0]

        if not batch:
            batch = num_data

        tot_batch = int(math.ceil(float(num_data) / float(batch)))
        if validate_data:
            for i in range(epoch):
                for j in range(tot_batch):
                    batch_X = self.__X[j*batch:min(num_data, (j+1)*batch), :]
                    batch_y = self.__y[j*batch:min(num_data, (j+1)*batch)]
                    self.__b, self.__wt = self._gd_update(
                        batch_X, batch_y, self.__wt, self.__b, rate, alpha)
                print 'Epoch {:5d}: err = {:.6f} validate = {:.6f}'.format(
                    i+1,
                    self.err_insample(),
                    self.err(validate_data[0], validate_data[1]))
        else:
            for i in range(epoch):
                for j in range(tot_batch):
                    batch_X = self.__X[j*batch:min(num_data, (j+1)*batch), :]
                    batch_y = self.__y[j*batch:min(num_data, (j+1)*batch)]
                    self.__b, self.__wt = self._gd_update(
                        batch_X, batch_y, self.__wt, self.__b, rate, alpha)
                print 'Epoch {:5d}: err = {:.6f}'.format(i+1, self.err_insample())

    def _gd_update(self, X, y, wt, b, rate, alpha):  # noqa: N803
        num_data = X.shape[0]

        # y_pred
        y_pred = np.sum(wt * X, axis=1) + b
        # y_pred - y
        y_diff = y_pred - y

        # b update
        b_gradient = np.sum(y_diff)/num_data
        new_b = b - rate * b_gradient
        # wt update
        new_wt = []
        for i, w in enumerate(wt):
            w_gradient = np.sum(X[:, i] * y_diff)/num_data + alpha * w
            new_wt.append(w - rate * w_gradient)
        new_wt = np.array(new_wt, dtype='float64')
        return (new_b, new_wt)

    def predict(self, X):  # noqa: N803
        if type(X) == list:
            X = np.array(X, dtype='float64')

        if X.shape[1] != self.__wt.shape[0]:
            raise ValueError('shape of input x does not match shape of weight')

        return np.sum(self.__wt * X, axis=1)+self.__b

    def err_insample(self):
        if self.__X.size == 0 or self.__y.size == 0:
            raise RuntimeError('in-sample data not found')

        return self.err(self.__X, self.__y)

    def err(self, X, y):  # noqa: N803
        self._check_data(X, y)

        if self.__wt.size == 0 or self.__b.size == 0:
            raise RuntimeError('you should train model first')
        y_pred = np.sum(self.__wt * X, axis=1) + self.__b

        err = (np.sum((y - y_pred)**2) / y.shape[0])**0.5
        return round(err, 8)

    def _check_data(self, X, y):  # noqa: N803
        if X.shape[0] != y.shape[0]:
            raise ValueError('shape of X and y do not match')
