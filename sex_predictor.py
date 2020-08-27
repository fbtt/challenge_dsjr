import os
import argparse
import catboost
import pandas as pd
from catboost import CatBoostClassifier


BASE_DIR = os.getcwd()
PATH_MODEL = os.path.join(BASE_DIR, 'catboost_model')


def preprocess(df_):
    """This function preprocess a dataframe so that it can be used as input of the model.
    
    This function drops 'cp' and 'slope', fill missing values in 'chol', substitutes cells
    with value 4 to 0 in 'ca' and converts cells with values 0 to 2 in 'thal'.

    Parameters
    ----------
    df_ : pd.DataFrame
        A pd.DataFrame to be preprocessed

    Returns
    -------
    pd.DataFrame
        A preprocessed pd.DataFrame
    """

    df_ = df_.copy()
    df_.drop('cp', axis='columns', inplace=True)
    df_.drop('slope', axis='columns', inplace=True)
    df_['chol'].fillna(239.0, inplace=True)
    df_.loc[df_[df_.ca==4].index, 'ca'] = 0
    df_.loc[df_[df_.thal==0].index, 'thal'] = 2

    return df_


def predict_sex(df_, clf_):
    """[summary]

    Parameters
    ----------
    df_ : pd.DataFrame
        DataFrame with the observations to be predict
    clf_ : CatBoostClassifier object
        A trained CatBoostClassifier

    Returns
    -------
    pd.DataFrame
        A pd.DataFrame containing the column 'sex' which has the predicted values
    """

    df_ = df_.copy()
    y_pred = clf_.predict(df_)
    y_pred = ['M' if k==0 else 'F' for k in y_pred]
    df_pred = pd.DataFrame(data=y_pred, columns=['sex'])
    
    return df_pred


def main():
    
    parser = argparse.ArgumentParser(description="This script predicts if a batch of patients are male or female")
    parser.add_argument('--input_file', required=True, help="input_file (csv) with the same structure of 'test_data_CANDIDATE.csv', but with an unknown number of lines (n)")
    args = parser.parse_args()
    input_file = args.input_file

    # load newsample.csv
    file_dir = os.path.join(BASE_DIR, input_file)
    df = pd.read_csv(file_dir, index_col='index')

    # preprocess
    df_preprocessed = preprocess(df)

    # load trained model
    clf = CatBoostClassifier()
    clf.load_model(PATH_MODEL)

    # predict
    df_predicted = predict_sex(df_preprocessed, clf)

    # save predictions
    df_predicted.to_csv('newsample_PREDICTIONS_Fernando_Battisti.csv', index=False)


if __name__ == '__main__':
    main()