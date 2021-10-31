"""
Offline grid search script for accelerated performances.
"""

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer

from preprocessing import remove_outlier


if __name__ == '__main__':
    pwd = os.path.dirname(os.path.realpath(__file__))
    tgt_wine = 'red'
    model_str = 'mlp'
    wine_path = f'{pwd}/__data__/winequality-{tgt_wine}.csv'
    wine_df = pd.read_csv(wine_path, delimiter=';')
    RANDOM_STATE = 42
    tgt_col = ['quality']
    feat_col = wine_df.columns[:-1]
    wine_df = remove_outlier(wine_df, feat_col)
    white_X, white_y = wine_df[feat_col].values, \
        wine_df[tgt_col].values

    # Creating test set
    # Stratify guarantees we have same proportion in train and test
    # than original dataset
    X_train, X_test, y_train, y_test = train_test_split(
        white_X, white_y, test_size=0.2, random_state=RANDOM_STATE, stratify=white_y)
    y_train, y_test = y_train.ravel(), y_test.ravel()

    # Creating validation set
    X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
    stop = len(X_train) // 4
    X_val, y_val = X_train[:stop], y_train[:stop]
    X_train, y_train = X_train[stop:], y_train[stop:]

    # Scaling data
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_val_sc = scaler.transform(X_val)

    if model_str == 'mlp':
        model = MLPClassifier()
        grid_parameters = {
            'hidden_layer_sizes': [(50, 50), (100, 50), (500, 15), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': 10.0 ** -np.arange(1, 10),
            'learning_rate': ['adaptive'],
            'max_iter': [5000]
        }
    elif model_str == 'svc':
        grid_parameters = {
            'kernel': ['linear', 'poly'],  # 'rbf'],
            'C': [0.001, 0.01, 0.1, 1, 2, 5, 10],
            'gamma': [0.01, 0.1, 0.5, 1]
        }
        model = SVC()

    clf = GridSearchCV(
        model,
        grid_parameters,
        n_jobs=-1,
        verbose=1,
        scoring=make_scorer(
            f1_score,
            average='weighted'))
    clf.fit(X_train_sc, y_train)
    joblib.dump(clf, f'{model_str}_{tgt_wine}_cv.pkl')
    print('param', clf.best_params_)
    print(clf.score(X_test_sc, y_test))
    print(clf.score(X_val_sc, y_val))
