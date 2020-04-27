import datetime as dt
import pandas as pd
import numpy as np
import requests
import io
import os

import sys
sys.path.append('cache')
import local_time as lt


# Constants
CACHE_PATH = '../cache/csv/'
CACHE = CACHE_PATH + 'power_measures_{}.csv'

MAX_DAYS = 31         # At most, Elia can return `MAX_DAYS` days of data
TIME_DELTA = 15 * 60  # Elia return one measure every 15 minutes

SHEET = {
    'solar': 'SolarForecasts',
    'wind': 'forecast',
}

URL = {  # regionId = 3 for Wallonia, sourceID = 8 for Liege Province
    'solar': 'https://griddata.elia.be/eliabecontrols.prod/interface/fdn/'
             'download/solarweekly/currentselection?dtFrom={}&dtTo={}'
             '&sourceID=8&isOffshore=&isEliaConnected=&forecast=solar',
    'wind': 'https://griddata.elia.be/eliabecontrols.prod/interface/fdn/'
            'download/windweekly/currentselection?dtFrom={}&dtTo={}&regionId=3'
            '&isOffshore=&isEliaConnected=&forecast=wind',
}

ELIA_COLUMNS = {
    'Week-ahead Forecast [MW]': 'week_ahead',
    'Day-ahead forecast (11h00) [MW]': 'day_ahead',
    'Most recent forecast [MW]': 'most_recent',
    'Measured & upscaled [MW]': 'measured',
    'Monitored Capacity [MW]': 'monitored_capacity',
    'Active Decremental Bids [yes/no]': 'active_decremental_bids',
    'DateTime': 'timestamp',
    'Day-Ahead forecast [MW]': 'day_ahead',
    'Week-Ahead forecast [MW]': 'week_ahead',
    'Real-time Upscaled Measurement [MW]': 'measured',
    'Corrected Upscaled Measurement [MW]': 'corrected',
    'Monitored Capacity [MWp]': 'monitored_capacity'
}

# Setup
os.makedirs(CACHE_PATH, exist_ok=True)


def request_elia_(power_source, ts_start, ts_end, verbose=False):
    '''
    Request the desired data to Elia, between the timestamps ts_start and
    ts_end, for `power_source`. Returns it in a pandas DataFrame.

    Arguments
    =========
     - power_source: string
        either `solar` or `wind`; if `solar`, request the solar data in Liege
        province, if `wind`, request the wind data in Wallonia
     - ts_start: int
        start timestamp, the corresponding date is included
     - ts_end: int
        end timestamp, the corresponding date is included
     - verbose: bool
        whether to comment on the requests results
    '''
    start = lt.date_from_ts(ts_start)
    end = lt.date_from_ts(ts_end)

    # Request the data to elia
    data = requests.get(URL[power_source].format(start, end))
    if data.status_code != 200:
        print('The {} data has not been retrieved between {} and {}'
              .format(power_source, start, end))
        return None

    if verbose:
        print('{} data successfully requested to elia bewteen {} and {}'
              .format(power_source.capitalize(), start, end))

    # Convert from Excel Sheet to DataFrame
    content = io.BytesIO(data.content)
    df = pd.read_excel(content, sheet_name=SHEET[power_source], header=3)

    # Clean the columns name and set index
    df.rename(columns=ELIA_COLUMNS, inplace=True)
    df['timestamp'] = df['timestamp'].apply(lt.dt_from_formatted_date,
                                            timestamp=True)

    return lt.set_time_index(df, 'timestamp', verbose=verbose)


def cache_measures_between(power_source, start, end=None, verbose=True):
    '''
    Cache the measures from elia between `start` and `end` for the considered
    power source. If end is None, cache until yesterday.

    Arguments
    =========
     - power_source: string
        either `solar` or `wind`, if `solar` request the solar data in Liege
        province, if `wind`, request the wind data in Wallonia
     - start: datetime.date
        start date (included)
     - end: datetime.date
        end date (included), should be before today
     - verbose: bool
        whether to comment on the number of non retrieved entries
    '''
    if power_source != 'wind' and power_source != 'solar':
        raise ValueError('`power_source` should be either `solar` or `wind`')

    # If end is None, cache until yesterday
    if end is None:
        end = dt.date.today() - dt.timedelta(days=1)

    if start > end or end >= dt.date.today():
        raise ValueError('`start` should be before `end`, `end` should be '
                         'before today')

    cache = CACHE.format(power_source)
    ts_start = lt.dt_from_date(start, end=False, timestamp=True)
    ts_end = lt.dt_from_date(end, end=True, timestamp=True)

    # Create the timestamp range index
    timestamp_range = range(ts_start, ts_end, TIME_DELTA)
    measures = pd.DataFrame(index=timestamp_range)

    # Recover what has already been cached
    if os.path.isfile(cache):
        cached = pd.read_csv(cache, index_col='timestamp')
        measures = pd.merge(cached, measures, left_index=True,
                            right_index=True, how='outer')

    # Recover what is missing
    missing = measures.index[measures.isna().all(axis=1)].values
    max_deviation = MAX_DAYS * 3600 * 24

    while True:
        if len(missing) == 0:
            break

        # Take the largest possible interval
        first = np.min(missing)
        last = np.max(missing[missing < (first + max_deviation)])

        # Request to elia
        new_data = request_elia_(power_source, first, last, verbose=verbose)

        # Update with the new values
        if new_data is not None:
            if len(measures.columns) == 0:
                measures[new_data.columns] = new_data
            else:
                measures.loc[new_data.index, new_data.columns] = new_data

        missing = missing[missing > last]

    # Cache the data
    measures.to_csv(cache, index=True, index_label='timestamp')

    # Comment on the non retrieved data
    n_missing = measures.isna().all(axis=1).sum()

    if verbose:
        print('{} data has been successfully cached'
              .format(power_source.capitalize()))
        if n_missing > 0:
            print('{} values have not been retrieved from Elia, consider '
                  'calling this again, hoping that Elia responds'
                  .format(n_missing))


def get_cached_measures(power_source, start=None, end=None):
    '''
    Returns a DataFrame for `power_source` constituted of the data cached using
    `cache_measures_between`. If `start` or `end` are outside of the cached
    data, the index is extended and the DataFrame contains na values.

    Arguments
    =========
     - power_source: string
        either `solar` or `wind`, if `solar` request the solar data in Liege
        province, if `wind`, request the wind data in Wallonia
     - start: datetime.date
        start date (included), if None, the first date of the cached data
     - end: datetime.date
        end date (included), if None, the last date of the cached data
    '''
    if power_source != 'wind' and power_source != 'solar':
        raise ValueError('`power_source` should be either `solar` or `wind`')

    cache = CACHE.format(power_source)
    if not os.path.isfile(cache):
        raise FileNotFoundError('The requested data has not been cached, '
                                'consider using `cache_measures_between`')

    cached = pd.read_csv(cache, index_col='timestamp')

    if start is None:
        ts_start = cached.index.min()
    else:
        ts_start = lt.dt_from_date(start, end=False, timestamp=True)
    if end is None:
        ts_end = cached.index.max() + TIME_DELTA
    else:
        ts_end = lt.dt_from_date(end, end=True, timestamp=True)

    # Extend the time index
    timestamp_range = range(ts_start, ts_end, TIME_DELTA)
    measures = pd.DataFrame(index=timestamp_range)

    measures[cached.columns] = cached

    return measures


def get_power_between(power_source, start=None, end=None):
    '''
    Request the measures or the forecast (`start` and `end` can be after today)
    from elia between `start` and `end` for the considered power source.

    If start and end are None, returns the data for today.

    Arguments
    =========
     - power_source: string
        either `solar` or `wind`, if `solar` request the solar data in Liege
        province, if `wind`, request the wind data in Wallonia
     - start: datetime.date
        day from which to request data
     - end: datetime.date
        day until which to request data
    '''
    if power_source != 'wind' and power_source != 'solar':
        raise ValueError('`power_source` should be either `solar` or `wind`')

    if start is None and end is None:
        ts_start = lt.dt_from_date(dt.date.today(), timestamp=True)
        ts_end = ts_start
    elif end is None:
        ts_start = lt.dt_from_date(start, timestamp=True)
        ts_end = ts_start
    else:
        ts_start = lt.dt_from_date(start, timestamp=True)
        ts_end = lt.dt_from_date(end, timestamp=True)

    return request_elia_(power_source, ts_start, ts_end, verbose=False)


if __name__ == '__main__':

    start_cache = dt.date(2018, 1, 1)

    cache_measures_between('solar', start_cache)
    cache_measures_between('wind', start_cache)
