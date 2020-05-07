# This file computes the mean distribution of the predictions for both the
# provincial and the panelwise model, for tomorrow's production (by default).
# Then, it combines the results in a csv results file.

import pandas as pd
import numpy as np
import os
import pytz

from datetime import datetime, timedelta, timezone, date, time
from cache.power import get_cached_measures, get_power_between
from solar_provincial import solar_provincial
from solar_panelwise import solar_panelwise

t, forecast_panelwise, lower_panelwise, upper_panelwise = solar_panelwise()
forecast_provincial, lower_provincial, upper_provincial, elia_forecast = solar_provincial()

# Write summary dataframe and add yesterday's measurements
if os.path.isfile('../products/csv/solar_forecast.csv'):
    results = pd.read_csv('../products/csv/solar_forecast.csv',
                          index_col='timestamp')
    results.index = pd.DatetimeIndex(results.index)

    yesterday = date.today() - timedelta(days=1)

    # Retrieve measurments of yesterday
    measured = None
    while measured is None:
        measured = get_power_between('solar', yesterday, yesterday)

    measured = measured['measured'].groupby(measured.index // 1800).mean()
    measured.index *= 1800
    measured.index = [datetime.fromtimestamp(x, tz=timezone.utc) for x in measured.index]
    measured.index = measured.index - timedelta(minutes=15)

    # isin
    isin = measured.index.isin(results.index)
    results.loc[measured[isin].index, 'measured'] = measured[isin]
else:
    results = pd.DataFrame(columns=['measured', 'forecast_panelwise', 'forecast_provincial', 'lower_provincial',
                                    'upper_provincial', 'elia_forecast'])

new_results = pd.DataFrame({
    'forecast_panelwise': forecast_panelwise,
    'forecast_provincial': forecast_provincial,
    'lower_provincial': lower_provincial,
    'upper_provincial': upper_provincial,
    'elia_forecast': elia_forecast
}, index=t)

results = results.append(new_results, sort=False)
results = results[~results.index.duplicated(keep='first')]

results.to_csv('../products/csv/solar_forecast.csv', index=True, index_label='timestamp')
