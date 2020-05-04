from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

import datetime as dt
import numpy as np
import pickle
import os

import sys
sys.path.append('cache')
import wind_ls

from quantile_models import QuantileExtraTrees, QuantileGradientBoosting


# Constants
CACHE_PATH = '../cache/pkl/'
CACHE = CACHE_PATH + '{}_{}.pkl'

# Setup
os.makedirs(CACHE_PATH, exist_ok=True)


def cache_model(train_set_only=False, model_name='qgb'):
    """
    Cache the QuantileGradientBoosting model. The cached model is either a
    model trained on the whole learning set, or a model trained on 70 % of this
    training set, leaving the rest available for testing.

    Arguments
    =========
     - train_set_only: bool
        whether to train on the train set only
    """
    X, y, t = wind_ls.get_learning_set()

    if train_set_only:
        train_set = 'train'
        X, _, y, _, t, _ = train_test_split(X, y, t, test_size=.3,
                                            random_state=0)
    else:
        train_set = 'all'

    if model_name == 'qxt':
        model = QuantileExtraTrees(.1, .9, 1000, n_jobs=-1,
                                   min_samples_split=10)
    elif model_name == 'qgb':
        model = QuantileGradientBoosting(.1, .9, 1000, subsample=.7)
    else:
        raise NotImplementedError('The SL method has not been implemented yet')

    model.fit(X, y)

    with open(CACHE.format(model_name, train_set), 'wb') as file:
        pickle.dump(model, file)

    print('Model successfully cached')


def get_model(train_set_only=False, model_name='qgb'):
    """
    Returns one of the cached models: either the model trained on the whole
    learning set, or the one trained on the train set.

    Arguments
    =========
     - train_set_only: bool
        whether to train on the train set only
    """
    if train_set_only:
        filename = CACHE.format(model_name, 'train')
    else:
        filename = CACHE.format(model_name, 'all')

    if not os.path.isfile(filename):
        raise FileNotFoundError('The requested model has not been cached, '
                                'consider using `cache_model`')

    with open(filename, 'rb') as file:
        models = pickle.load(file)

    return models


if __name__ == '__main__':

    parser = ArgumentParser(description='Cache the trained Quantile model')
    parser.add_argument('model', type=str, choices=['qxt', 'qgb'], 
                        help='name of the model to cache')
    parser.add_argument('--train-set-only', action='store_true',
                        help='Train an alternative model on a train set '
                             'composed of 0.7 of the learning set, for '
                             'evaluation purpose')
    args = parser.parse_args()

    cache_model(train_set_only=args.train_set_only, model_name=args.model)
