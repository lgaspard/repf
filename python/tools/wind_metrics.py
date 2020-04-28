from sklearn.metrics import mean_absolute_error

import numpy as np

import sys
sys.path.append('cache')
import wind_ls


# Constants
MAX_POWER = np.max(wind_ls.get_learning_set()[1])


def MAE(y_true, y_pred):
    """
    Returns the mean aboslute error bewteen `y_true` and `y_pred`.

    Arguments
    =========
     - y_true: numpy.array
        true values
     - y_pred: numpy.array
        predicted values
    """
    return mean_absolute_error(y_true, y_pred)


def nMAE(y_true, y_pred):
    """
    Returns the normalized mean aboslute error bewteen `y_true` and `y_pred`.
    Normalized means that it is divided by the maximum values observed on the 
    true data.

    Arguments
    =========
     - y_true: numpy.array
        true values
     - y_pred: numpy.array
        predicted values
    """
    return MAE(y_true, y_pred) / MAX_POWER


def MQL(y_true, q_pred, a):
    """
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
    """
    multiplier = [a if y < q else 1 - a for y, q in zip(y_true, q_pred)]
    return np.mean(multiplier * np.abs(y_true - q_pred))


def performance_summary(model, X_test, y_test):
    """
    Print a string that summarizes the MAE, nMAE, and MQL for the upper and
    lower quantiles.

    Arguments
    =========
     - model: sklearn.base.BaseEstimator with `predict` method
        should have parameters `lower_quantile` and `upper_quantile`, and
        returns the prediction under the format (lower, pred, upper)
     - X_test: numpy.array
        the test set features
     - y_test: numpy.array
        the test set true values
    """
    model_name = type(model).__name__
    print(model_name + '\n' + '=' * len(model_name))

    params = model.get_params()
    lower_q = params['lower_quantile']
    upper_q = params['upper_quantile']

    y_lower, y_pred, y_upper = model.predict(X_test)
    
    print('MAE  : {:.2f} MW\n'.format(MAE(y_test, y_pred)) +
          'nMAE : {:.2f} %\n'.format(100 * nMAE(y_test, y_pred)) +
          'MQL10: {:.2f} MW\n'.format(MQL(y_test, y_lower, lower_q)) +
          'MQL90: {:.2f} MW\n'.format(MQL(y_test, y_upper, upper_q)))
