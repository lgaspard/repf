import datetime as dt
import pandas as pd
import numpy as np
import os

import cache.power
import cache.wind_ls
import cache.wind_model
from cache.wind_model import QuantileGradientBoosting

import matplotlib.pyplot as plt
import tools.plot_setup

from argparse import ArgumentParser

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Constants
PLOT_PATH = '../products/pdf/'
RESULTS_PATH = '../products/csv/'
RESULTS_FILE = RESULTS_PATH + 'wind_forecast.csv'

# Setup
os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def create_daily_forecast():
    """
    Cache a history of daily forecast made at midnight for the following day
    (24 hours to 48 hours later). Furthermore, adds the measurements of the
    past days to the history. Should be called automatically each day in order
    to provide a complete daily forecast history.
    """
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)
    yesterday = today - dt.timedelta(days=1)

    # Retrieve weather predictions
    w, t, f = cache.wind_ls.get_forecasting_set(tomorrow, tomorrow,
                                                elia_forecast=True)

    # Forecasting
    qgb = cache.wind_model.get_model()
    y_lower, y_pred, y_upper = qgb.predict(w)

    # Save results and add measurments to the previous forecasts
    if os.path.isfile(RESULTS_FILE):

        results = pd.read_csv(RESULTS_FILE, index_col='timestamp')

        # Retrieve measurments of yesterday
        measured = cache.power.get_power_between('wind', yesterday, yesterday)
        measured = measured['measured'].groupby(measured.index // 3600).mean()
        measured.index *= 3600

        if set(measured.index).issubset(set(results.index)):
            results.loc[measured.index, 'measured'] = measured
    else:
        results = pd.DataFrame(columns=['measured', 'forecast', 'lower',
                                        'upper', 'elia_forecast'])

    new_results = pd.DataFrame({
        'forecast': y_pred,
        'lower': y_lower,
        'upper': y_lower,
        'elia_forecast': f
    }, index=t)

    results = results.append(new_results, sort=False)
    results = results[~results.index.duplicated(keep='first')]

    results.to_csv(RESULTS_FILE, index=True, index_label='timestamp')


if __name__ == '__main__':

    parser = ArgumentParser(description='Forecast the wind power production')
    parser.add_argument('model', type=str, choices=['qxt', 'qgb'], 
                        help='name of the model to evaluate')
    parser.add_argument('--display', action='store_true',
                        help='Whether to display the plot')
    args = parser.parse_args()

    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)

    # Retrieve weather predictions
    w, t, f = cache.wind_ls.get_forecasting_set(today, tomorrow,
                                                elia_forecast=True)

    # Forecasting
    model = cache.wind_model.get_model(model_name=args.model)
    y_lower, y_pred, y_upper = model.predict(w)

    # Total energy produced
    lower_energy = np.trapz(y_lower.reshape((-1, 24)), axis=1)
    mean_energy = np.trapz(y_pred.reshape((-1, 24)), axis=1)
    upper_energy = np.trapz(y_upper.reshape((-1, 24)), axis=1)
    print('Lower bound on energy:', lower_energy)
    print('Predicted energy:', mean_energy)
    print('Upper bound on energy:', upper_energy)

    # Plot
    fig, ax = plt.subplots()
    time = np.vectorize(dt.datetime.fromtimestamp)(t)
    ax.plot(time, y_pred, label='Forecast')
    ax.plot(time, f, label='Elia Forecast')
    ax.fill_between(time, y_lower, y_upper, alpha=.3)
    ax.set_xlabel('Time [CEST]')
    ax.set_ylabel('Power [MW]')
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plot_filename = 'forecasted_on_{}.pdf'.format(today)
    plt.savefig(PLOT_PATH + plot_filename, transparent=True)
    if args.display:
        plt.show()
