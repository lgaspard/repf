import datetime as dt
import numpy as np
import os

import cache.wind_ls
import cache.wind_model
from cache.wind_model import QuantileGradientBoosting

from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
from tools.wind_metrics import *

import matplotlib.pyplot as plt
import tools.plot_setup

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Constants
PLOT_PATH = '../products/pdf/'

# Setup
os.makedirs(PLOT_PATH, exist_ok=True)


def sort_all(array, *arrays):
    """
    Sorts all array and all the arrays in `arrays` according to the order of 
    `array`.

    Arguments
    =========
     - array: numpy.array
        array to use as ordering array
     - arrays: list of numpy.array
        arrays to sort in the same order as `array`
    """
    indexes = np.argsort(array)
    return (array[indexes],) + tuple(a[indexes] for a in arrays)


if __name__ == '__main__':

    parser = ArgumentParser(description='Evaluate the cached wind model')
    parser.add_argument('model', type=str, choices=['qxt', 'qgb'], 
                        help='name of the model to evaluate')
    parser.add_argument('--forecast', action='store_true',
                        help='Evaluate on weather prediction (test set) '
                             'instead of validation set')
    parser.add_argument('--residuals', action='store_true',
                        help='Plot the residuals of the predictions')
    args = parser.parse_args()

    if args.forecast:
        filename = 'wind_eval_{}_test.pdf'.format(args.model)
        X, y, t, f = cache.wind_ls.get_test_set(elia_forecast=True)
        model = cache.wind_model.get_model(train_set_only=False,
                                           model_name=args.model)
    else:
        filename = 'wind_eval_{}_valid.pdf'.format(args.model)
        X, y, t, f = cache.wind_ls.get_learning_set(elia_forecast=True)
        _, X, _, y, _, t, _, f = train_test_split(X, y, t, f, test_size=.3,
                                                  random_state=0, shuffle=False)
        model = cache.wind_model.get_model(train_set_only=True,
                                           model_name=args.model)

    # Extraction of the parameters and metrics computations
    params = model.get_params()
    lower_q = params['lower_quantile']
    upper_q = params['upper_quantile']

    y_lower, y_pred, y_upper = performance_summary(model, X, y,
                                                   return_pred=True)

    # Plot the results
    fig, ax = plt.subplots()

    datetime_vec = np.vectorize(dt.datetime.fromtimestamp)
    time = datetime_vec(t)
    time, y_pred, y_lower, y_upper, y, f = sort_all(
        time, y_pred, y_lower, y_upper, y, f)

    if args.residuals:
        ax.plot(time, np.abs(y_pred - y), label='Forecast residuals')
        ax.plot(time, np.abs(f - y), label='Elia forecast residuals')
        filename = 'res_' + filename
    else:
        ax.plot(time, y_pred, label='Forecast')
        ax.plot(time, y, label='Elia measures')
        ax.plot(time, f, label='Elia forecast')
        ax.fill_between(time, y_lower, y_upper, alpha=.3)

    print('Elia\'s MAE: {:.2f}'.format(MAE(f, y)))

    ax.legend()
    ax.set_xlabel('Time [CEST]')
    ax.set_ylabel('Power absolute error [MW]')
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(PLOT_PATH + filename, transparent=True)
    plt.show()
