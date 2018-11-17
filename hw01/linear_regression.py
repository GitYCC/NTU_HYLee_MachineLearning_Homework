#!python2.7
import os
from collections import OrderedDict
from datetime import timedelta

import numpy as np
import pandas as pd

from gradient_descent import GradientDescent

PATH_GIVEN_TRAIN = './given/train.csv'
PATH_GIVEN_TEST = './given/test_X.csv'
PATH_TRAIN_DATA = './train_data.csv'
PATH_VALID_DATA = './valid_data.csv'
PATH_RESULT = 'result/linear_regression.csv'


def _create_tidy_training_set():
    # read raw data and label it
    # 1...24 mean hour of that day
    colnames = ['Date', 'Site', 'Item'] + map(str, range(24))
    df = pd.read_csv(PATH_GIVEN_TRAIN, names=colnames, skiprows=1)

    # remove column 'Site'
    df = df.loc[:, ['Date', 'Item'] + map(str, range(24))]

    # snapshot:
    # colums: Date, Item, 1...24(hour)

    # melt 'Hour' to column
    df = pd.melt(df,
                 id_vars=['Date', 'Item'],
                 value_vars=map(str, range(24)),
                 var_name='Hour',
                 value_name='Value')

    # snapshot:
    # colums: Date, Item, Hour, Value

    # generate 'Datetime'
    df['Datetime'] = pd.to_datetime(df.Date + ' ' + df.Hour + ':00:00')
    df = df.loc[:, ['Datetime', 'Item', 'Value']]

    # snapshot:
    # colums: Datetime, Item, Value

    # replace NR to 0
    df.loc[df.Value == 'NR', 'Value'] = 0

    # change 'Value' type
    df['Value'] = df['Value'].astype(float)

    # pivot 'Item' to columns
    df = df.pivot_table(values='Value', index='Datetime', columns='Item', aggfunc='sum')

    # snapshot:
    # index: Datetime
    # colums: AMB_TEMP, CH4, CO,..., PM2.5,..., WS_HR

    return df


def _split_df_into_train_and_valid(df):
    df_valid = df.loc[df.index.month == 12, :]
    df_train = df.loc[df.index.month != 12, :]
    return (df_train, df_valid)


def _convert_to_problem_regression_form(df):
    """Convert to the problem regression form.

    The problem is that using parameters among continue 9 hours to predict PM2.5 on next hour.

    Args:
        df (pd.Dataframe): `index` is datetime
                           `columns` include parameters AMB_TEMP, CH4, CO,..., PM2.5,..., WS_HR

    """
    data = OrderedDict()

    # create columns of `data`
    param_list = df.columns.tolist()
    for i in range(9):
        for param in param_list:
            data['{:02d}h__{}'.format(i+1, param)] = []
    data['10h__PM2.5'] = []

    # add content into `data`
    datetime_list = df.index
    d1h = timedelta(hours=1)
    for m in pd.unique(datetime_list.month):
        for timestamp in (df.loc[df.index.month == m, :]).index:
            start, end = timestamp, timestamp + 9*d1h
            sub_df = df.loc[(start <= df.index) & (df.index <= end), :]
            if sub_df.shape[0] == 10:
                for i in range(9):
                    for param in param_list:
                        data['{:02d}h__{}'.format(i+1, param)].append(
                            sub_df.loc[timestamp+i*d1h, param])
                data['10h__PM2.5'].append(sub_df.loc[timestamp + 9 * d1h, 'PM2.5'])

    return pd.DataFrame(data)


def preprocess_training_set():
    if os.path.isfile(PATH_TRAIN_DATA) and os.path.isfile(PATH_VALID_DATA):
        train_df = pd.read_csv(PATH_TRAIN_DATA)
        valid_df = pd.read_csv(PATH_VALID_DATA)
    else:
        df = _create_tidy_training_set()
        tmp_train_df, tmp_valid_df = _split_df_into_train_and_valid(df)

        train_df = _convert_to_problem_regression_form(tmp_train_df)
        train_df.to_csv(PATH_TRAIN_DATA, index=None)
        valid_df = _convert_to_problem_regression_form(tmp_valid_df)
        valid_df.to_csv(PATH_VALID_DATA, index=None)

    train_X = np.array(train_df.loc[:, train_df.columns != '10h__PM2.5'])
    train_y = np.array(train_df.loc[:, '10h__PM2.5'])

    valid_X = np.array(valid_df.loc[:, valid_df.columns != '10h__PM2.5'])
    valid_y = np.array(valid_df.loc[:, '10h__PM2.5'])

    col_name = (train_df.loc[:, train_df.columns != '10h__PM2.5']).columns

    return (train_X, train_y, valid_X, valid_y, col_name)


def preprocess_testing_set(col_name):
    col_names = ['ID', 'Item'] + map(lambda x: '{:02d}h'.format(x), range(1, 10))
    test_df = pd.read_csv(PATH_GIVEN_TEST, names=col_names, header=None)

    # snapshot:
    # columns: ID, Item, 01h, 02h, ..., 09h

    # replace NR to 0
    for col in map(lambda x: '{:02d}h'.format(x), range(1, 10)):
        test_df.loc[(test_df.Item == 'RAINFALL') & (test_df[col] == 'NR'), col] = 0

    # ['ID','Item','Hour','Value'] form
    test_df = test_df.pivot_table(index=['ID', 'Item'], aggfunc='sum')
    test_df = test_df.stack()
    test_df = test_df.reset_index()
    test_df.columns = ['ID', 'Item', 'Hour', 'Value']

    # snapshot:
    # columns: ID, Item, Hour, Value

    # combine 'Hour' and 'Item' to 'Col'
    test_df['Col'] = test_df.Hour + '__' + test_df.Item
    test_df = test_df[['ID', 'Col', 'Value']]

    # snapshot:
    # columns: ID, Col, Value

    # pivot 'Col' to columns
    test_df = test_df.pivot_table(values='Value', index='ID', columns='Col', aggfunc='sum')

    # snapshot:
    # index: ID
    # columns: 01h__AMB_TEMP, 01h__CH4, 01h__CO,..., 09h__WS_HR

    # re-order
    test_df['ID_Num'] = test_df.index.str.replace('id_', '').astype('int')
    test_df = test_df.sort_values(by='ID_Num')
    test_df = test_df.drop('ID_Num', axis=1)

    # snapshot:
    # index: ID
    # columns: ID_Num, 01h__AMB_TEMP, 01h__CH4, 01h__CO,..., 09h__WS_HR

    test_X = np.array(test_df[col_name], dtype='float64')
    ids = np.array(test_df.index, dtype='str')
    return (test_X, ids)


def main():
    train_X, train_y, valid_X, valid_y, col_name = preprocess_training_set()

    # ### gradient descent
    gd = GradientDescent()

    # method: pseudo_inverse
    gd.train_by_pseudo_inverse(train_X, train_y, alpha=0.5, validate_data=(valid_X, valid_y))

    # method: gradient desecent
    # init_wt = gd.wt
    # init_b  = gd.b
    #
    # gd.train(train_X,train_y,epoch=10,rate=0.000001,batch=100,alpha=0.00000001,
    #    init_wt=np.array(init_wt),init_b=init_b,
    #    validate_data = (valid_X,valid_y))

    # ### predict
    test_X, ids = preprocess_testing_set(col_name)
    pred_y = gd.predict(test_X)

    result = list()
    for i in range(ids.shape[0]):
        result.append([ids[i], pred_y[i]])

    # output
    with open(PATH_RESULT, 'w') as fw:
        for id, pred in result:
            fw.write('{id},{pred}\n'.format(id=id, pred=pred))

if __name__ == '__main__':
    main()
