from argparse import ArgumentParser

import datetime as dt
import pandas as pd
import requests
import os

import sys
sys.path.append('cache')
import local_time as lt
import wind_farms


# Constants
CACHE_PATH = '../cache/wcsv/'
CACHE_MEASURES = CACHE_PATH + 'weather_measures_{:.4f}_{:.4f}.csv'
CACHE_FORECAST = CACHE_PATH + 'weather_forecast_{:.4f}_{:.4f}.csv'

API_URL = 'https://api.darksky.net/forecast'

INFOS = {
    'time': 'timestamp',
    'windSpeed': 'wind_speed',
    'windGust': 'wind_gust',
    'windBearing': 'wind_bearing',
    'temperature': 'temperature',
    'humidity': 'humidity',
    'pressure': 'pressure'
}

# The first 6 keys are reserved for the caching, the last is free for forecast
# or measures requested on live in other codes
API_KEYS = [
    '2fbc6f0f10436c2b3c5d77db97cc0d72',
    'd71a76b28cc3722db0c12a09e0f63c91',
    '6edc8381a4498ef53f9da6beec22e4da',
    '2086ce711b3d45ad5942cf3538a9e39b',
    '5e5672d724f0a52fea0cc072a3b66ebf',
    'e8f7287827ed3347e7efa4fc5503eca8',
    '03947f1b45e68b144618a7691cad26f0'
]
CALLS_PER_KEY = 1000

# Setup
os.makedirs(CACHE_PATH, exist_ok=True)


def air_density_(T, h, p):
    """
    Compute the theoretic air density from the temperature, humidity fraction,
    and pressure. The air density is returned in kg/m3.

    Arguments
    =========
     - T: float
        temperature in Kelvin
     - h: float
        humidity fraction (between 0. and 1.)
     - p: pressure
        pressure in Pascal
    """
    m_water = 0.018016  # water molar mass (kg/mol)
    m_air = 0.0289654   # water molar mass (kg/mol)
    R = 8.314           # universal gas constant (J/(K mol))

    T_C = T - 273.15         # temperature (°C)
    press_sat = 6.102 * 10 ** ((7.5 * T_C) / (T_C + 237.8))  # sat. vapor (Pa)
    press_w = h * press_sat  # partial pressure of water vaport (Pa)
    press_a = p - press_w    # partial pressure dry air (Pa)

    return (press_a * m_air + press_w * m_water) / (R * T)


def request_darksky_(loc, date, api_key, verbose=True):
    """
    Request the weather infos in `INFOS` and the air density to darksky, for
    the 24 hours of the day `date`. Returns the results in a pandas DataFrame.

    Arguments
    =========
     - loc: 2-tuple
        location of the measure under the format (lat, lon)
     - date: datetime.date
        date for which the weather infos are requested
     - api_key: string
        a valid API key for the darkski API
     - verbose: bool
        whether to print errors and successful calls
    """
    # Generate request URL
    ts = lt.dt_from_date(date, end=False, timestamp=True)
    lat, lon = loc
    lat = round(lat, 4)
    lon = round(lon, 4)
    url = '{}/{}/{},{},{}'.format(API_URL, api_key, lat, lon, ts)

    # Request to the API
    data = requests.get(url).json()
    if verbose:
        print('[API USAGE] Weather requested on {} at ({:.4f}, {:.4f})'
              .format(date, lat, lon))

    if 'error' in data:
        raise IOError(data['error'])

    if 'hourly' not in data and verbose:
        print('[ERROR] Not available for {}, ({:.4f}, {:.4f})'
              .format(date, lat, lon))
        return None

    # DataFrame creation
    results = pd.DataFrame({
        info: [measures[info] if info in measures else float('NaN')
               for measures in data['hourly']['data']] for info in INFOS
    })

    # Change of units and creation of the new info `air_density`
    def f_to_k(f): return (f - 32) / 1.8 + 273.15  # from farheneit to celsius
    results['temperature'] = results['temperature'].apply(f_to_k)
    results['pressure'] *= 100  # from HPa to Pa
    results['air_density'] = (results[['temperature', 'humidity', 'pressure']]
                              .apply(lambda x: air_density_(*x), axis=1))
    results.rename(columns=INFOS, inplace=True)
    results.set_index('timestamp', inplace=True)

    # Creation of index for all hours in day
    weather = pd.DataFrame(index=range(ts, ts + 3600 * 24, 3600))
    weather[results.columns] = results

    return weather


def cache_measures_between(loc, start, end, api_keys=API_KEYS[:-1],
                           verbose=True):
    """
    Cache the weather measures between `start` and `end` at the location `loc`.

    Arguments
    =========
     - loc: 2-tuple
        location of the measure under the format (lat, lon)
     - start: datetime.date
        starting date
     - start: datetime.date
        end date
     - api_key: list of string
        a list of valid API key for the darkski API
     - verbose: bool
        whether to print errors and successful calls
    """
    if start > end or end >= dt.date.today():
        raise ValueError('`start` should be before `end`, `end` should be '
                         'before today')
    k = 0
    lat, lon = loc
    cache = CACHE_MEASURES.format(lat, lon)
    ts_start = lt.dt_from_date(start, end=False, timestamp=True)
    ts_end = lt.dt_from_date(end, end=True, timestamp=True)

    # Create the timestamp range index
    timestamp_range = range(ts_start, ts_end, 3600)
    measures = pd.DataFrame(index=timestamp_range)

    # Recover what has already been cached
    if os.path.isfile(cache):
        cached = pd.read_csv(cache, index_col='timestamp')
        measures = pd.merge(cached, measures, left_index=True,
                            right_index=True, how='outer')

    # Recover what is missing
    missing = measures.index[measures.isna().all(axis=1)]
    missing = missing[(missing >= ts_start) & (missing < ts_end)]
    missing_days = missing.map(lt.date_from_ts).drop_duplicates()

    for day in missing_days:
        while True:
            try:
                new_data = request_darksky_(loc, day, api_keys[k],
                                            verbose=verbose)
                break
            except IOError as e:
                print('[ERROR]', e)
                k += 1
            except IndexError as e:
                print('[ERROR] No more API key for today')
                return k

        if new_data is not None:
            if len(measures.columns) == 0:
                measures[new_data.columns] = new_data
            else:
                measures.loc[new_data.index, new_data.columns] = new_data

    # Cache the data
    measures.to_csv(cache, index=True, index_label='timestamp')

    # Comment on the non retrieved data
    n_missing = measures.loc[missing].isna().all(axis=1).sum()

    if verbose:
        print('Data successfully cached for ({:.4f}, {:.4f})'.format(lat, lon))
        if n_missing > 0:
            print(n_missing, 'values have not been retrieved from Darksky')

    return k


def cache_all_measures_between(start, end=None, api_keys=API_KEYS[:-1],
                               verbose=True):
    """
    Cache the weather measures between `start` and `end` at all the locations
    in the cached wind farms locations.

    Arguments
    =========
     - start: datetime.date
        starting date
     - start: datetime.date
        end date
     - end: datetime.date
        end date, if None, cache until yesterday
     - api_key: list of string
        a list of valid API key for the darkski API
     - verbose: bool
        whether to print errors and successful calls
    """
    batch_size = 30
    locs = wind_farms.get_farms_loc()

    if end is None:
        end = dt.date.today() - dt.timedelta(days=1)

    k = 0
    while start <= end:
        current_end = min([end, start + dt.timedelta(days=batch_size)])
        for loc in zip(locs['lat'], locs['lon']):
            k = cache_measures_between(loc, start, current_end, api_keys,
                                       verbose=True)
            api_keys = api_keys[k:]
            if len(api_keys) == 0:
                return

        if verbose:
            print('Data retrieved for all locations bewteen {} and {}'
                  .format(start, current_end))

        start = current_end + dt.timedelta(days=1)


def get_cached_measures(loc, start=None, end=None):
    """
    Returns a DataFrame for location `loc` constituted of the cached data
    using `cache_measures_between`. If `start` or `end` are outside of the
    cached data, the index is extended and the DataFrame contains na values.

    Arguments
    =========
     - loc: 2-tuple
        location of the measure under the format (lat, lon)
        province, if `wind`, request the wind data in Wallonia
     - start: datetime.date
        start date (included), if None, the first date of the cached data
     - end: datetime.date
        end date (included), if None, the last date of the cached data
    """
    lat, lon = loc
    cache = CACHE_MEASURES.format(lat, lon)
    if not os.path.isfile(cache):
        raise FileNotFoundError('The requested data has not been cached, '
                                'consider using `cache_measures_between`')

    cached = pd.read_csv(cache, index_col='timestamp')

    if start is None:
        ts_start = cached.index.min()
    else:
        ts_start = lt.dt_from_date(start, end=False, timestamp=True)
    if end is None:
        ts_end = cached.index.max() + 3600
    else:
        ts_end = lt.dt_from_date(end, end=True, timestamp=True)

    timestamp_range = range(ts_start, ts_end, 3600)
    measures = pd.DataFrame(index=timestamp_range)

    measures[cached.columns] = cached

    return measures


def cache_forecast_tomorrow(loc, api_key=API_KEYS[-1], verbose=True):
    """
    Cache the weather forecast for tomorrow at the location `loc`. Should be
    called automatically each day at the same time of the day in order to
    constitute constitent dataset of weather forecastings.

    Arguments
    =========
     - loc: 2-tuple
        location of the measure under the format (lat, lon)
     - api_keys: string
        a valid API key for the darkski API
     - verbose: bool
        whether to comment on the cached data
    """
    lat, lon = loc
    cache = CACHE_FORECAST.format(lat, lon)
    if os.path.isfile(cache):
        forecast = pd.read_csv(cache, index_col='timestamp')
    else:
        forecast = pd.DataFrame()

    tomorrow = dt.date.today() + dt.timedelta(days=1)
    forecast = forecast.append(get_weather_between(loc, tomorrow))

    forecast = forecast.loc[~forecast.index.duplicated(keep='first')]
    forecast.to_csv(cache, index=True, index_label='timestamp')
    if verbose:
        print('Forecast has been successfully cached for ({:.4f}, {:.4f})'
              .format(lat, lon))


def cache_all_forecast_tomorrow(api_key=API_KEYS[:-1], verbose=True):
    """
    Cache the weather forecast for all wind farms locations from the cached
    locations.

    Arguments
    =========
     - api_key: string
        a valid API key for the darkski API
     - verbose: bool
        whether to comment on the cached data
    """
    locs = wind_farms.get_farms_loc()

    for loc in zip(locs['lat'], locs['lon']):
        cache_forecast_tomorrow(loc, api_key, verbose=verbose)

    if verbose:
        print('Data retrieved for all locations for tomorrow')


def get_cached_forecast(loc):
    """
    Returns the cached weather forecasts for the location `loc`.

    Arguments
    =========
     - loc: 2-tuple
        location of the measure under the format (lat, lon)
    """
    lat, lon = loc
    cache = CACHE_FORECAST.format(lat, lon)
    if not os.path.isfile(cache):
        raise FileNotFoundError('The requested data has not been cached, '
                                'consider using `cache_measures_between`')

    return pd.read_csv(cache, index_col='timestamp')


def get_weather_between(loc, start=None, end=None, api_key=API_KEYS[-3],
                        verbose=False):
    """
    Request the measures or the forecast (`start` and `end` can be after today)
    from darksky between `start` and `end` for the considered power source.

    If start and end are None, returns the data for today.

    Arguments
    =========
     - loc: 2-tuple
        location of the measure under the format (lat, lon)
     - start: datetime.date
        day from which to request data
     - end: datetime.date
        day until which to request data
     - api_key: string
        a valid API key for the darkski API
     - verbose: bool
        whether to comment on the cached data
    """
    if start is None and end is None:
        start = dt.date.today()
        end = start
    elif end is None:
        end = start

    weather = pd.DataFrame()
    while start <= end:
        new_data = request_darksky_(loc, start, api_key=api_key,
                                    verbose=verbose)
        weather = weather.append(new_data)
        start += dt.timedelta(days=1)

    return weather


if __name__ == '__main__':

    def date_parser(s): return dt.datetime.strptime(s, '%Y-%m-%d').date()

    parser = ArgumentParser(description='Fill the measures cache between START'
                                        ' and END, or add tomorrow forecast to'
                                        ' the forecast cache')
    parser.add_argument('--forecast', action='store_true',
                        help='Cache the forecast for tomorrow instead of the '
                             'measures')
    parser.add_argument('--start', default=dt.date(2018, 1, 1),
                        help='Start date of the measures cache',
                        type=date_parser)
    parser.add_argument('--end', default=None, type=date_parser,
                        help='End date of the measures cache')
    args = parser.parse_args()

    if args.forecast:
        cache_all_forecast_tomorrow(verbose=True)
    else:
        cache_all_measures_between(args.start, args.end, verbose=True)
