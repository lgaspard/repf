from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import os
import re

import sys
sys.path.append('cache')


# Constants
WT_MODELS = '../resources/csv/wt_models.csv'
WT_LIST = '../resources/csv/wt_list.csv'

CACHE_PATH = '../cache/csv/'
CACHE_WT_SPECS = CACHE_PATH + 'wt_specs.csv'
CACHE_FARMS_LOC = CACHE_PATH + 'wt_farms_loc.csv'

RE_WIND_SPEED = r'data:\s\{\slabels:\s?\[([\d,\.]+)\]'
RE_POWER = (r'datasets:\s\[\{"label":"power \(kW\)","yAxisID":"power",'
            r'"data":\s?\[([\d",\.]+)\]')

INDEX_COL = ['brand', 'type', 'nominal_power']

# Setup
os.makedirs(CACHE_PATH, exist_ok=True)


def nb_from_speed_(x):
    return float(x.get_text().strip().strip('m/s').strip())


def cache_wt_specs():
    """
    Cache the wind turbines specifications available from the website
    `wind-turbines-models.com` using the list of wind turbines models in
    wallonia and the appropriate to the data sheet on the website.
    """
    if os.path.isfile(CACHE_WT_SPECS):

        # Retrieve what has already been cached
        wt_models = pd.read_csv(CACHE_WT_SPECS, index_col=INDEX_COL)
        required_cols = ['cutin_wspd', 'cutout_wspd', 'rated_wspd']
        to_request = wt_models[required_cols].isna().any(axis=1)

        # Get the references from the list of wind turbines models
        refs = pd.read_csv(WT_MODELS, index_col=INDEX_COL)['reference']
        wt_models['reference'] = refs

    else:
        # Create a new DataFrame from the list of wind turbines models
        wt_models = pd.read_csv(WT_MODELS, index_col=INDEX_COL)
        to_request = wt_models.index

    for index, url in wt_models.loc[to_request, 'reference'].items():

        response = requests.get(url)

        wt_model_name = ' '.join(map(str, index))
        print('Collecting data for {}'.format(wt_model_name))

        # Parse the reference page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extraction of cut-in, cut-out and rated wind speed
        table_container = soup.find('div', {'data-tabname': 'leistung'})
        table = table_container.find('table')

        cutin = cutout = rated = None

        try:
            cutin = nb_from_speed_(table.findAll('tr')[2].findAll('td')[1])
            cutout = nb_from_speed_(table.findAll('tr')[4].findAll('td')[1])
            rated = nb_from_speed_(table.findAll('tr')[3].findAll('td')[1])
        except ValueError:
            pass

        if cutin:
            wt_models.loc[index, 'cutin_wspd'] = cutin
        if cutout:
            wt_models.loc[index, 'cutout_wspd'] = cutout
        if rated:
            wt_models.loc[index, 'rated_wspd'] = rated

        # Extraction of the power curve if any
        canvas = soup.find('canvas')
        if canvas is None:
            continue

        # Retrieve raw wind curves
        div_script_canvas = canvas.find_parent().find_parent()
        script = div_script_canvas.find('script').get_text()

        wind_speed = re.search(RE_WIND_SPEED, script).group(1)
        power = re.search(RE_POWER, script).group(1)

        # Clean the data
        wind_speed = wind_speed.split(',')
        wind_speed = [float(p) for p in wind_speed]
        power = power.split(',')
        power = [float(p.strip('"')) for p in power]

        power = np.trim_zeros(power, 'b')
        wind_speed = wind_speed[:len(power)]

        wt_models.loc[index, 'pc_wspd'] = ','.join([str(ws) for ws in
                                                    wind_speed])
        wt_models.loc[index, 'pc_power'] = ','.join([str(p) for p in power])

    # Cache
    wt_models.drop(columns=['reference', 'note'], inplace=True,
                   errors='ignore')
    wt_models.to_csv(CACHE_WT_SPECS)

    print('Wind turbines specifications successfully cached')


def get_wt_specs():
    """
    Returns the cached wind turbines characteristics.
    """
    if not os.path.isfile(CACHE_WT_SPECS):
        raise FileNotFoundError('The requested data has not been cached, '
                                'consider using `cache_wt_specs`')

    return pd.read_csv(CACHE_WT_SPECS, index_col=INDEX_COL)


def cache_farms_loc():
    """
    Cache the wind farms locations using the list of wind turbines in wallonia.
    """
    # List of all wind turbines
    cols = ['power_plant', 'lat', 'lon']
    wt_list = pd.read_csv(WT_LIST, usecols=cols)

    # Aggregation by power plants, taking the mean for latitute, and longitude
    wt_farms = wt_list.groupby(['power_plant']).mean()

    # Rounding of the latitude and longitude
    def round_4(x): return round(x, 4)
    wt_farms_loc = wt_farms[['lat', 'lon']].apply(round_4)

    # Cache
    wt_farms_loc.to_csv(CACHE_FARMS_LOC)

    print('Wind farms locations successfully cached')


def get_farms_loc():
    """
    Returns the cached wind farms locations.
    """
    if not os.path.isfile(CACHE_FARMS_LOC):
        raise FileNotFoundError('The requested data has not been cached, '
                                'consider using `cache_farms_loc`')

    return pd.read_csv(CACHE_FARMS_LOC, index_col='power_plant')


if __name__ == '__main__':

    cache_wt_specs()
    cache_farms_loc()
