import requests
import json
import pandas as pd
import numpy as np

from datetime import datetime, timezone, timedelta


def clean_missing_values(irradiance, start, end):
    time_check = pd.date_range(start=start, end=end, freq='30T')
    for d in time_check:
        if d not in irradiance.index:
            start = d - timedelta(minutes=30)
            end = d +  timedelta(minutes=30)

            missing = [d]
            missing_ts = [datetime.timestamp(d)]
            while end not in irradiance.index:
                missing.append(end)
                missing_ts.append(datetime.timestamp(end))
                end += timedelta(minutes=30)

            x = [datetime.timestamp(start), datetime.timestamp(end)]
            y = [irradiance.loc[start]['GHI'], irradiance.loc[end]['GHI']]

            interp = np.interp(missing_ts, x, y)

            for i in range(len(interp)):
                df = pd.DataFrame(data=[interp[i]], index=[missing[i]], columns=['GHI'])
                irradiance = pd.concat([irradiance, df], sort=True)

    irradiance.sort_index(inplace=True)
    return irradiance

def cache_irradiance_measures():
    ### CSV File
    obs_file = '../cache/csv/observed_irradiance.csv'

    headers = {'Accept': 'application/json'}
    params = {'api_key': 'Tk_DeBkkIWX1b3uHVe0L65nax9dQjfy7'}

    url_obs = 'https://api.solcast.com.au/weather_sites/c49a-0be9-87fd-bba1/estimated_actuals?format=json'

    ### Retrieve live data
    r = requests.get(url_obs, headers=headers, params=params)
    data = r.json()['estimated_actuals']

    idx = pd.DatetimeIndex([obs['period_end'] for obs in reversed(data)])
    idx = idx - timedelta(minutes=15)
    df = pd.DataFrame(data=[obs['ghi'] for obs in reversed(data)], index=idx, columns=['GHI'])

    cleaned_irr = clean_missing_values(df, start=idx[0], end=idx[-1])

    ### Retrieve previously collected observed irradiance
    obs_irr = pd.read_csv(obs_file, index_col=0)
    obs_irr.index = pd.DatetimeIndex(obs_irr.index)

    ### Concatenate both sources
    irradiance = pd.concat([obs_irr, cleaned_irr])
    irradiance = irradiance.loc[~irradiance.index.duplicated(keep='first')]

    ### Save observed irradiance data
    irradiance.to_csv(obs_file)
    print("Observed irradiance has been successfully retrieved.")

def cache_irradiance_forecasts():
    ### CSV File
    for_file = '../cache/csv/forecast_irradiance.csv'

    headers = {'Accept': 'application/json'}
    params = {'api_key': 'Tk_DeBkkIWX1b3uHVe0L65nax9dQjfy7'}

    url_for = 'https://api.solcast.com.au/weather_sites/c49a-0be9-87fd-bba1/forecasts?format=json'
    
    ### Retrieve forecast data
    r = requests.get(url_for, headers=headers, params=params)
    data = r.json()['forecasts']

    idx = pd.DatetimeIndex([obs['period_end'] for obs in data])
    idx = idx - timedelta(minutes=15)
    df = pd.DataFrame(data=[obs['ghi'] for obs in data], index=idx, columns=['GHI'])

    cleaned_irr = clean_missing_values(df, start=idx[0], end=idx[-1])

    ### Retrieve previously collected forecasted irradiance
    for_irr = pd.read_csv(for_file, index_col=0)
    for_irr.index = pd.DatetimeIndex(for_irr.index)

    ### Concatenate both sources
    irradiance = pd.concat([for_irr, cleaned_irr])
    irradiance = irradiance.loc[~irradiance.index.duplicated(keep='last')]

    ### Save forecasted irradiance data
    irradiance.to_csv(for_file)
    print("Forecast irradiance has been successfully retrieved.")

if __name__ == '__main__':
    cache_irradiance_measures()
    cache_irradiance_forecasts()