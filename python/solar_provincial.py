import numpy as np
import pandas as pd
import tools.plot_setup
import matplotlib.pyplot as plt

import pystan
import pytz

from datetime import datetime, timedelta, timezone, date, time
from cache.power import get_cached_measures, get_power_between
from cache.solar_weather import cache_irradiance_measures
from cache.solar_weather import cache_irradiance_forecasts


def pystan_model(obs_power, peak_area, eta, obs_irradiance, for_irradiance):
    model = """
    data {
      int<lower=0> n_obs; 
      vector[n_obs] power; 
      vector[n_obs] I_obs;  
      
      real<lower=0> area; 
      real<lower=0> efficiency; 

      int<lower=0> n_forecast; 
      vector[n_forecast] I_forecast; 
    }

    parameters {
      real<lower=0, upper=0.2> eta;
      real<lower=0> peak_area;
      real kVA;
      
      vector<lower=0>[n_obs] I;  // True measures (not observed)
      vector<lower=0>[n_forecast] I_for;  // True forecasts (not observed)
    }

    model {
      eta ~ normal(efficiency, 0.01);
      peak_area ~ normal(area, 1);
      
      I_obs ~ normal(I, 0.001);
      I_forecast ~ normal(I_for, 0.001);
      
      power ~ normal(kVA * peak_area * I * eta, 1e5);
    }

    generated quantities {
        vector[n_forecast] for_power; // Forecast output power

        for (i in 1:n_forecast) {
            for_power[i] = normal_rng(kVA * peak_area * I_for[i] * eta, 1e2);
        }
    }
    """

    # All data is passed as kW quantities (power is the data to fit to)
    dat = {'n_obs': obs_power.shape[0],
           'power': obs_power,
           'I_obs': obs_irradiance,
           'area': peak_area,
           'efficiency': eta,
           'n_forecast': for_irradiance.shape[0],
           'I_forecast': for_irradiance}

    sm = pystan.StanModel(model_code=model)

    return sm.sampling(data=dat, iter=30000, n_jobs=1, 
                       control={'adapt_delta': 0.95})


def get_cached_irradiance(start, end, start_for, end_for):
    # Data is in UTC. Forecast data starts on April 2nd (10:15)
    irradiance_observed = pd.read_csv(
        '../cache/csv/observed_irradiance.csv', index_col=0)
    irradiance_observed.index = pd.to_datetime(
        irradiance_observed.index, utc=True)

    irradiance_forecast = pd.read_csv(
        '../cache/csv/forecast_irradiance.csv', index_col=0)
    irradiance_forecast.index = pd.to_datetime(
        irradiance_forecast.index, utc=True)

    # Slice appropriate period
    obs_irradiance = irradiance_observed[start:end]
    for_irradiance = irradiance_forecast[start_for:end_for]

    return obs_irradiance, for_irradiance


def solar_provincial():

    # Read panels data and extract data about Liège (province)
    solar_stats = pd.read_csv('../resources/csv/solar_power_stats.csv', 
                              sep=';', thousands=".", index_col=0)
    kVA = solar_stats['kVA']
    ETA = 0.15

    # Squared meters of PV panels per kWPeak in Belgium
    # Source: https://energieplus-lesite.be/concevoir/photovoltaique/
    #                 predimensionner-l-installation/
    AREA_PEAK = 7

    # Average area of panels in the province of Liège
    A = (AREA_PEAK * kVA).sum()

    # Datetime definitions
    tz = pytz.timezone('UCT')
    tomorrow = date.today() + timedelta(days=1)

    START_FOR = datetime.combine(tomorrow, time())
    START_FOR = tz.localize(START_FOR)
    END_FOR = START_FOR + timedelta(days=1)

    START = START_FOR - timedelta(days=8)
    END = START_FOR - timedelta(days=1)

    # Retrieve cached irradiance data
    obs_irradiance, for_irradiance = get_cached_irradiance(
        START, END, START_FOR, END_FOR)

    # Check whether cached data was up to date
    n_obs = (END - START).days * 48

    if n_obs != obs_irradiance.shape[0]:
        print("Irradiance measures were not up to date. Caching...")
        cache_irradiance_measures()
        cache_irradiance_forecasts()
        obs_irradiance, for_irradiance = get_cached_irradiance(
            START, END, START_FOR, END_FOR)
    else:
        print("Irradiance measures were up to date.")
        print("Caching latest irradiance forecasts...")
        cache_irradiance_forecasts()
        obs_irradiance, for_irradiance = get_cached_irradiance(
            START, END, START_FOR, END_FOR)

    # Retrieve Elia's production data
    elia = get_power_between('solar', START, END_FOR)
    while elia is None:
        elia = get_power_between('solar', START, END_FOR)

    elia = elia.groupby(by=elia.index // 1800).mean()
    elia.index *= 1800
    elia.index = np.apply_along_axis(lambda l: [datetime.fromtimestamp(
        x, tz=timezone.utc) for x in l], 0, elia.index)

    elia['corrected'] = elia['corrected'].astype(float)
    elia['most_recent'] = elia['most_recent'].astype(float)

    # Divide Elia's production data into training data and forecast data and 
    # check for NaN
    train_elia = elia[(elia.index >= START) & (elia.index < END)].copy()
    if train_elia.isnull().values.any():
        train_elia.interpolate(inplace=True)

    elia_forecast = elia[(elia.index >= START_FOR) &
                         (elia.index < END_FOR)].copy()
    if elia_forecast.isnull().values.any():
        elia_forecast.interpolate(inplace=True)

    # Compute prior output power (MW)
    naive_power_obs = ETA * obs_irradiance * A / 1e6
    naive_power_for = ETA * for_irradiance * A / 1e6

    # Fit PyStan model on Elia's measures
    fit = pystan_model(train_elia['corrected'] * 1e3, AREA_PEAK, ETA,
                       obs_irradiance['GHI'] / 1e3, for_irradiance['GHI'] / 1e3)
    post_samples = fit.extract()

    # Retrieve predicted power and compute mean distribution
    # post_samples['for_power'] is in kW
    forecast_power = post_samples['for_power'] / 1e3
    mu_for = np.mean(forecast_power, axis=0)
    std_for = np.std(forecast_power, axis=0)

    # Plot Elia's forecast against posterior predictive model
    forecast_time_series = pd.date_range(start=for_irradiance.index[0], 
                                         end=for_irradiance.index[-1], 
                                         freq='30T', 
                                         closed=None)

    fig, ax = plt.subplots()
    ax.fill_between(forecast_time_series, mu_for - std_for,
                    mu_for + std_for, color="b", alpha=0.1)
    ax.plot(forecast_time_series, mu_for, color="b")
    ax.plot(elia_forecast.index, elia_forecast['most_recent'], color="orange")

    ax.set_xlabel('Time')
    ax.set_ylabel('Output power (MW)')
    fig.autofmt_xdate()
    ax.legend(['Posterior forecast +- 3std', "Elia's forecast"])

    ax.grid()
    fig.tight_layout()
    fig.savefig('../products/pdf/solar_provincial.pdf', transparent=True)

    return mu_for, mu_for - 3 * std_for, mu_for + 3 * std_for, elia_forecast['most_recent'].values


if __name__ == '__main__':

    # Computes the mean distribution of the posterior predicted power, along 
    # with +- 3std for tomorrow. For the sake of convenience, we also return 
    # Elia's forecast for the considered period.
    # To compute for another date, change 'tomorrow' assignment.
    solar_provincial()
