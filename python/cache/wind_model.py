from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

from argparse import ArgumentParser

import datetime as dt
import numpy as np
import pickle
import os

import sys
sys.path.append('cache')
import wind_ls


# Constants
CACHE_PATH = '../cache/pkl/'
CACHE_MODEL_ALL = CACHE_PATH + 'qgb_all.pkl'
CACHE_MODEL_TRAIN = CACHE_PATH + 'qgb_train.pkl'

# Setup
os.makedirs(CACHE_PATH, exist_ok=True)


class QuantileGradientBoosting(BaseEstimator):
    """
    Wrapper for 3 Gradient Boosting Estimators:
     - self.lower: a quantile gradient boosting for the lower quantile
     - self.upper: a quantile gradient boosting for the upper quantile
     - self.mean: a least-square gradient boosting
    """

    def __init__(self, n_estimators, lower_quantile, upper_quantile,
                 learning_rate=.1):

        super(QuantileGradientBoosting, self).__init__()

        self.n_estimators = n_estimators
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.learning_rate = learning_rate

        self.lower = GradientBoostingRegressor(loss='quantile',
                                               alpha=lower_quantile,
                                               n_estimators=n_estimators,
                                               learning_rate=learning_rate)
        self.upper = GradientBoostingRegressor(loss='quantile',
                                               alpha=upper_quantile,
                                               n_estimators=n_estimators,
                                               learning_rate=learning_rate)
        self.mean = GradientBoostingRegressor(loss='ls', criterion='mse',
                                              n_estimators=n_estimators,
                                              learning_rate=learning_rate)

    def fit(self, X, y):
        '''
        Fits the three estimators to the data.
        '''
        self.lower.fit(X, y)
        self.upper.fit(X, y)
        self.mean.fit(X, y)

    def predict(self, X):
        '''
        Returns the prediction of the three estimators in a 3-tuple: lower, 
        mean, and upper.
        '''
        l = self.lower.predict(X)
        u = self.upper.predict(X)
        m = self.mean.predict(X)
        return l, m, u

    def get_params(self, deep=False):
        params = {
            'n_estimators': self.n_estimators,
            'lower_quantile': self.lower_quantile,
            'upper_quantile': self.upper_quantile,
            'learning_rate': self.learning_rate
        }
        if deep:
            params = {**params, **self.lower.get_params(),
                      **self.upper.get_params(), **self.mean.get_params()}
        return params

    def set_params(self, n_estimators=None, lower_quantile=None,
                   upper_quantile=None, learning_rate=None):
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if lower_quantile is not None:
            self.lower_quantile = lower_quantile
        if upper_quantile is not None:
            self.upper_quantile = upper_quantile
        if learning_rate is not None:
            self.learning_rate = learning_rate


def cache_model(train_set_only=False):
    '''
    Cache the QuantileGradientBoosting model. The cached model is either a
    model trained on the whole learning set, or a model trained on 70 % of this
    training set, leaving the rest available for testing.

    Arguments
    =========
     - train_set_only: bool
        whether to train on the train set only
    '''
    X, y, t = wind_ls.get_learning_set()

    if train_set_only:
        filename = CACHE_MODEL_TRAIN
        X, _, y, _, t, _ = train_test_split(X, y, t, test_size=.3,
                                            random_state=0)
    else:
        filename = CACHE_MODEL_ALL

    qgb = QuantileGradientBoosting(500, .1, .9, learning_rate=.1)
    qgb.fit(X, y)

    with open(filename, 'wb') as file:
        pickle.dump(qgb, file)

    print('Model successfully cached')


def get_model(train_set_only=False):
    '''
    Returns one of the cached models: either the model trained on the whole
    learning set, or the one trained on the train set.

    Arguments
    =========
     - train_set_only: bool
        whether to train on the train set only
    '''
    if train_set_only:
        filename = CACHE_MODEL_TRAIN
    else:
        filename = CACHE_MODEL_ALL

    if not os.path.isfile(filename):
        raise FileNotFoundError('The requested model has not been cached, '
                                'consider using `cache_model`')

    with open(filename, 'rb') as file:
        models = pickle.load(file)

    return models


if __name__ == '__main__':

    parser = ArgumentParser(description='Cache the trained Quantile Gradient '
                                        'model')
    parser.add_argument('--train-set-only', action='store_true',
                        help='Train an alternative model on a train set '
                             'composed of 0.7 of the learning set, for '
                             'evaluation purpose')
    args = parser.parse_args()

    cache_model(train_set_only=args.train_set_only)
