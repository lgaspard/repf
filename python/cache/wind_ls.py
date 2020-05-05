from argparse import ArgumentParser
import datetime as dt
import pandas as pd
import numpy as np
import os

import sys
sys.path.append('cache')
import power
import wind_farms
import wind_weather


# Constants
CACHE_PATH = '../cache/csv/'
CACHE_LS = CACHE_PATH + 'learning_set.csv'
CACHE_TS = CACHE_PATH + 'test_set.csv'
PLOT_PATH = '../products/pdf/'

# Setup
os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)


def cache_learning_set(show_na=True, test_set=False, end=None,
                       variables=['wind_speed', 'wind_gust']):
    """
    Cache the learning set using the cached weather data and cached wind power
    data.

    Arguments
    =========
     - show_na: bool
        whether to generate a plot to visualize the missingness
     - test_set: bool
        whether to generate the test set with the cached weather forecast
        instead of the learning set with the cached weather measures
    """
    elia = power.get_cached_measures('wind', end=end)
    elia = elia[['measured', 'day_ahead']]

    # Aggregation of the elia data to each hour, as the weather is hourly
    elia = elia.groupby(by=elia.index // 3600).mean()
    elia.index *= 3600

    locs = wind_farms.get_farms_loc()

    # Request the cached weather data for each location (measures or forecast)
    weather = pd.DataFrame()
    for lat, lon in zip(locs['lat'], locs['lon']):
        if test_set:
            new_data = wind_weather.get_cached_forecast((lat, lon))
        else:
            new_data = wind_weather.get_cached_measures((lat, lon))
        weather = pd.concat((weather, new_data[variables]), axis=1)

    learning_set = pd.concat((weather, elia), axis=1)

    # Generate and save the missingness visualization
    if show_na:
        sys.path.append('.')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import tools.plot_setup

        fig, ax = plt.subplots()
        ax.imshow(learning_set.isna(), aspect='auto', cmap=cm.cividis)
        ax.set_xlabel('Variables')
        ax.set_ylabel('Samples')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.tight_layout()
        plt.savefig(PLOT_PATH + 'na.pdf')
        plt.show()

    # Drop rows with missing values
    n_na = learning_set.isna().any(axis=1).sum()
    learning_set.dropna(inplace=True)
    print('There were {} na values in the learning set'.format(n_na))

    # Cache
    if test_set:
        learning_set.to_csv(CACHE_TS, index=True, index_label='timestamp')
    else:
        learning_set.to_csv(CACHE_LS, index=True, index_label='timestamp')

    print('The learning set has been successfully cached')


def get_learning_set(test_set=False, elia_forecast=False):
    """
    Returns the cached learning set in numpy arrays: X, y, t, and optionally 
    f, the forecast of elia.

    Arguments
    =========
     - test_set: bool
        whether to get the test set with the cached weather forecast instead
        of the learning set with cached weather measures
     - elia_forecast: bool
        whether to return elia forecast alongside
    """
    if not os.path.isfile(CACHE_LS):
        raise FileNotFoundError('The requested data has not been cached, '
                                'consider using `cache_learning_set`')

    if test_set:
        learning_set = pd.read_csv(CACHE_TS, index_col='timestamp')
    else:
        learning_set = pd.read_csv(CACHE_LS, index_col='timestamp')

    X = learning_set.values[:, :-2]
    y = learning_set.values[:, -2]
    t = learning_set.index.values
    f = learning_set.values[:, -1]

    if elia_forecast:
        return X, y, t, f
    return X, y, t


def cache_test_set():
    """
    Wrapper to cache the test set with weather forecast instead of the
    learning set with weather measures.
    """
    cache_learning_set(show_na=False, test_set=True)


def get_test_set(elia_forecast=False):
    """
    Wrapper to get the test set with weather forecast instead of the learning
    set with weather measures.

    Arguments
    =========
     - elia_forecast: bool
        whether to return elia forecast alongside
    """
    return get_learning_set(test_set=True, elia_forecast=elia_forecast)


def get_forecasting_set(start, end, elia_forecast=False,
                        variables=['wind_speed', 'wind_gust']):
    """
    Returns the inputs, and optionally the forecast of elia, in order to
    produce forecast based on this input.

    Arguments
    =========
     - start: datetime.date
        start date (included)
     - end: datetime.date
        end date (included)
     - elia_forecast: bool
        whether to return elia forecast alongside
    """
    elia = power.get_power_between('wind', start, end)['most_recent']

    # Aggregation of the elia data to each hour, as the weather is hourly
    elia = elia.groupby(by=elia.index // 3600).mean()
    elia.index *= 3600

    # Reassemble all weather measurements
    weather = pd.DataFrame()
    locs = wind_farms.get_farms_loc()

    for lat, lon in zip(locs['lat'], locs['lon']):

        new_data = wind_weather.get_weather_between((lat, lon), start, end,
                                                    verbose=True)
        # Use only relevant variables
        weather = pd.concat((weather, new_data[variables]), axis=1)

    learning_set = pd.concat((weather, elia), axis=1)
    learning_set.dropna(inplace=True)

    X = learning_set.values[:, :-1]
    f = learning_set.values[:, -1]
    t = learning_set.index

    if elia_forecast:
        return X, t, f
    return X, t


if __name__ == '__main__':

    parser = ArgumentParser(description='Cache the learning set or test set')
    parser.add_argument('--forecast', action='store_true',
                        help='Cache the forecast instead of weather test set')
    args = parser.parse_args()

    if args.forecast:
        cache_test_set()
    else:
        cache_learning_set(end=dt.date(2020, 3, 31))
