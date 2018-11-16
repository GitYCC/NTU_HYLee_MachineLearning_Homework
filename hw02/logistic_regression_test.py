#!python2.7
import pandas as pd
import numpy as np
from logistic_regression_train import LogisticRegression

COLUMNS = [
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
        'Feature_capital_run_length_total', 'label'
        ]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='HW2: Logistic Regression Testing')
    parser.add_argument('-m',  metavar='MODEL',  type=str,  nargs='?',
                        help='path of input model',  required=True)
    parser.add_argument('-data',  metavar='Test_DATA',  type=str,  nargs='?',
                        help='path of testing data',  required=True)
    parser.add_argument('-csv',  metavar='OUTPUT',  type=str,  nargs='?',
                        help='path of output csv',  required=True)

    args = parser.parse_args()

    data = args.data
    model = args.m
    out_csv = args.csv

    lg = LogisticRegression.load(model)

    df = pd.read_csv(data, names=COLUMNS)
    X_test = np.array(df.drop(['data_id'], axis=1))

    df['label'] = np.argmax(lg.predict(X_test), axis=1)

    df[['data_id', 'label']].to_csv(out_csv, index=None)


if __name__ == '__main__':
    main()
