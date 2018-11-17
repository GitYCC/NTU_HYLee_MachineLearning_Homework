import pandas as pd
import numpy as np

TEST_COLUMNS = [
        'data_id', 'Feature_make', 'Feature_address', 'Feature_all', 'Feature_3d',
        'Feature_our', 'Feature_over', 'Feature_remove', 'Feature_internet', 'Feature_order',
        'Feature_mail', 'Feature_receive', 'Feature_will', 'Feature_people', 'Feature_report',
        'Feature_addresses', 'Feature_free', 'Feature_business', 'Feature_email', 'Feature_you',
        'Feature_credit', 'Feature_your', 'Feature_font', 'Feature_000', 'Feature_money',
        'Feature_hp', 'Feature_hpl', 'Feature_george', 'Feature_650', 'Feature_lab',
        'Feature_labs', 'Feature_telnet', 'Feature_857', 'Feature_data', 'Feature_415',
        'Feature_85', 'Feature_echnology', 'Feature_1999', 'Feature_parts', 'Feature_pm',
        'Feature_direct', 'Feature_cs', 'Feature_meeting', 'Feature_original', 'Feature_project',
        'Feature_re', 'Feature_edu', 'Feature_table', 'Feature_conference', 'Feature_;',
        'Feature_(', 'Feature_[', 'Feature_!', 'Feature_$', 'Feature_#',
        'Feature_capital_run_length_average', 'Feature_capital_run_length_longest',
        'Feature_capital_run_length_total'
        ]

COLUMNS = TEST_COLUMNS + ['label']

PATH_SPAM_TRAIN = './given/spam_train.csv'
PATH_SPAM_TEST = './given/spam_test.csv'

VALID_RATIO = 0.8


def get_train_set():
    df = pd.read_csv(PATH_SPAM_TRAIN, names=COLUMNS)
    X = np.array(df.drop(['data_id', 'label'], axis=1))
    y = np.hstack(
        (np.array(df[['label']]), 1 - np.array(df[['label']]))
    )

    num_data = X.shape[0]
    num_train = int(VALID_RATIO * num_data)

    X_train = X[0:num_train, :]
    y_train = y[0:num_train, :]

    X_valid = X[num_train:, :]
    y_valid = y[num_train:, :]

    return (X_train, y_train, X_valid, y_valid)


def get_test_set():
    df = pd.read_csv(PATH_SPAM_TEST, names=TEST_COLUMNS)
    ids = df['data_id']
    df = df.drop(['data_id'], axis=1)
    X_test = np.array(df)
    return (X_test, ids)
