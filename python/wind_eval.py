import datetime as dt
import numpy as np
import os

import cache.wind_ls
import cache.wind_model
from cache.wind_model import QuantileGradientBoosting

import matplotlib.pyplot as plt
import tools.plot_setup

from argparse import ArgumentParser

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Constants
PLOT_PATH = '../products/pdf/'
MAX_POWER = np.max(cache.wind_ls.get_learning_set()[1])

# Setup
os.makedirs(PLOT_PATH, exist_ok=True)


def MAE(y_true, y_pred):
    '''
    Returns the mean aboslute error bewteen `y_true` and `y_pred`.

    Arguments
    =========
     - y_true: numpy.array
        true values
     - y_pred: numpy.array
        predicted values
    '''
    return mean_absolute_error(y_true, y_pred)


def nMAE(y_true, y_pred):
    '''
    Returns the normalized mean aboslute error bewteen `y_true` and `y_pred`.
    Normalized means that it is divided by the maximum values observed on the 
    true data.

    Arguments
    =========
     - y_true: numpy.array
        true values
     - y_pred: numpy.array
        predicted values
    '''
    return MAE(y_true, y_pred) / MAX_POWER


def MQL(y_true, q_pred, a):
    '''
    Returns the mean quantile loss for `a`-quantile `y_pred` and true value
    `y_true`.

    Arguments
    =========
     - y_true: numpy.array
        true values
     - q_pred: numpy.array
        predicted quantile
     - a: float
        quantile level
    '''
    multiplier = [a if y < q else 1 - a for y, q in zip(y_true, q_pred)]
    return np.mean(multiplier * np.abs(y_true - q_pred))


def model_performance(y, y_lower, y_pred, y_upper, lower_q, upper_q):
    '''
    Returns a string that summarizes the MAE, nMAE, and MQL for the upper and
    lower quantiles.

    Arguments
    =========
     - y_true: numpy.array
        true values
     - y_lower: numpy.array
        predicted lower quantile
     - y_pred: numpy.array
        predicted values
     - y_upper: numpy.array
        predicted upper quantile
     - lower_q: float
        lower quantile level
     - upper_q: float
        upper quantile level
    '''
    return ('MAE  : {:.2f} MW\n'.format(MAE(y, y_pred)) +
            'nMAE : {:.2f} %\n'.format(100 * nMAE(y, y_pred)) +
            'MQL10: {:.2f} MW\n'.format(MQL(y, y_lower, lower_q)) +
            'MQL90: {:.2f} MW'.format(MQL(y, y_upper, upper_q)))


def sort_all(array, *arrays):
    '''
    Sorts all array and all the arrays in `arrays` according to the order of 
    `array`.

    Arguments
    =========
     - array: numpy.array
        array to use as ordering array
     - arrays: list of numpy.array
        arrays to sort in the same order as `array`
    '''
    indexes = np.argsort(array)
    return (array[indexes],) + tuple(a[indexes] for a in arrays)


if __name__ == '__main__':

    parser = ArgumentParser(description='Evaluate the cached wind model')
    parser.add_argument('--forecast', action='store_true',
                        help='Evaluate on weather prediction (test set) '
                             'instead of validation set')
    args = parser.parse_args()

    if args.forecast:
        filename = 'eval_forecast_test.pdf'
        X, y, t, f = cache.wind_ls.get_test_set(elia_forecast=True)
        qgb = cache.wind_model.get_model(train_set_only=False)
    else:
        filename = 'eval_forecast_valid.pdf'
        X, y, t, f = cache.wind_ls.get_learning_set(elia_forecast=True)
        _, X, _, y, _, t, _, f = train_test_split(X, y, t, f, test_size=.3,
                                                  random_state=0)
        qgb = cache.wind_model.get_model(train_set_only=True)

    y_lower, y_pred, y_upper = qgb.predict(X)

    # Extraction of the parameters and metrics computations
    params = qgb.get_params()
    lower_q = params['lower_quantile']
    upper_q = params['upper_quantile']

    print(model_performance(y, y_lower, y_pred, y_upper, lower_q, upper_q))

    # Plot the results
    fig, ax = plt.subplots()

    datetime_vec = np.vectorize(dt.datetime.fromtimestamp)
    time = datetime_vec(t)
    time, y_pred, y_lower, y_upper, y, f = sort_all(
        time, y_pred, y_lower, y_upper, y, f)

    ax.plot(time, y_pred, label='Forecast')
    ax.plot(time, y, label='Elia measures')
    ax.plot(time, f, label='Elia forecast')

    ax.fill_between(time, y_lower, y_upper, alpha=.3)
    ax.legend()
    ax.set_xlabel('Time [UTC]')
    ax.set_ylabel('Power [MW]')
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(PLOT_PATH + filename, transparent=True)
    plt.show()
