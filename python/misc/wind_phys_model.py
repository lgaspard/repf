from argparse import ArgumentParser

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import sys
sys.path.append('.')
import tools.plot_setup
import cache.wind_farms
import cache.wind_ls
import cache.wind_weather
import cache.local_time

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Constants
WT_LIST = '../resources/csv/wt_list.csv'


def get_power_curve(cutin, cutout, rated, pc_wspd, pc_power, wt_power):

    # If a power curve is available, interpolation
    if isinstance(pc_wspd, str) and isinstance(pc_power, str):

        pc_wspd = [float(s) for s in pc_wspd.split(',')]
        pc_power = [float(p) / 1000 for p in pc_power.split(',')]

        p = interp1d(pc_wspd, pc_power, kind='cubic',
                     fill_value='extrapolate')

        end_curve = max(pc_wspd)

        return np.vectorize(
            lambda w: (0 if w <= cutin or w >= cutout else
            (wt_power if w >= end_curve else p(w)))
        )

    # If no power curve, creation of a naive theoretical curve
    else:
        constant = wt_power / (rated ** 3)

        end_curve = rated

        return np.vectorize(
            lambda w: (0 if w <= cutin or w >= cutout else
            (wt_power if w >= end_curve else constant * w ** 3))
        )


def aggregated_power_curves():

    power_curves = dict()

    wt_specs = cache.wind_farms.get_wt_specs()

    # Median imputation
    cutin_median = wt_specs['cutin_wspd'].median()
    wt_specs['cutin_wspd'].fillna(cutin_median, inplace=True)
    cutout_median = wt_specs['cutout_wspd'].median()
    wt_specs['cutout_wspd'].fillna(cutout_median, inplace=True)
    rated_median = wt_specs['rated_wspd'].median()
    wt_specs['rated_wspd'].fillna(rated_median, inplace=True)

    for index, row in wt_specs.iterrows():


        cutin = row['cutin_wspd']
        cutout = row['cutout_wspd']
        rated = row['rated_wspd']

        pc_wspd = row['pc_wspd']
        pc_power = row['pc_power']

        wt_power = index[2]

        power_curves[index] = get_power_curve(cutin, cutout, rated, pc_wspd,
                                              pc_power, wt_power)

    return power_curves


if __name__ == '__main__':

    def date_parser(s): return dt.datetime.strptime(s, '%Y-%m-%d').date()

    parser = ArgumentParser(description='Use the retrieved power curves and '
                                        'the cached weather data to produce '
                                        'wind prediction using physical '
                                        'modeling')
    parser.add_argument('--power-curves', action='store_true',
                        help='outputs to the power curves plot')
    parser.add_argument('--start', default=dt.date(2020, 3, 2),
                        help='Start date of the measures cache',
                        type=date_parser)
    parser.add_argument('--end', default=dt.date(2020, 3, 31),
                        help='End date of the measures cache',
                        type=date_parser)
    args = parser.parse_args()

    # Retrieve the data set
    start = args.start
    start_ts = cache.local_time.dt_from_date(start, end=False, timestamp=True)
    end = args.end
    end_ts = cache.local_time.dt_from_date(end, end=True, timestamp=True)

    X, y, t = cache.wind_ls.get_learning_set()
    indexes = (t >= start_ts) & (t < end_ts)
    X, y, t = X[indexes, :], y[indexes], t[indexes]

    time = list(map(dt.datetime.fromtimestamp, t))

    # Generate or interpolate the power curve data for each wind turbine model 
    power_curves = aggregated_power_curves()

    # Plot the power curves
    if args.power_curves:
        for model_ref, model in power_curves.items():
            ref = [str(x) for x in model_ref]

            fig, ax = plt.subplots()
            ax.set_xlabel('Wind Speed [m/s]')
            ax.set_ylabel('Power [MW]')
            ax.set_title(' '.join(ref))
            wind = np.arange(0, 30, .05)
            power = [model(w) for w in wind]
            ax.plot(wind, power)
            plt.tight_layout()
            plt.savefig('../products/pdf/pc_{}.pdf'.format('_'.join(ref)),
                        transparent=True)
            plt.close(fig)

    # Compute the total power for the considered date range
    power = 0
    farms_loc = cache.wind_farms.get_farms_loc()
    wt_list = pd.read_csv(WT_LIST)

    # Iterate over all wind turbines
    for index, row in wt_list.iterrows():

        wt_ref = row['brand'], row['type'], row['nominal_power']

        # Get the farm location associated with the current wind turbine
        farm_loc = farms_loc.loc[row['power_plant']]
        loc = (farm_loc['lat'], farm_loc['lon'])

        # Retrieve the wind speed at this location
        weather = cache.wind_weather.get_cached_measures(loc, start=start,
                                                         end=end)
        wind_speed = weather['wind_speed']

        # Compute power output for this wind turbine
        model = power_curves[wt_ref]
        power += model(wind_speed)

    time_power = list(map(dt.datetime.fromtimestamp, weather.index))

    # Plots the results
    fig, ax = plt.subplots()
    ax.plot(time_power, power, label='Physical forecast')
    ax.plot(time, y, label='Elia measures')

    ax.legend()
    ax.set_xlabel('Time [CEST]')
    ax.set_ylabel('Power [MW]')
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig('../products/pdf/phys_forecast.pdf', transparent=True)
    plt.show()
