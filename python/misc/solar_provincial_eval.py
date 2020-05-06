import numpy as np
import pandas as pd
import plot_setup
import matplotlib.pyplot as plt

import pystan

import sys
sys.path.append('.')

from datetime import datetime, timedelta, timezone
from cache.power import get_cached_measures, get_power_between


def compute_metrics(true, prediction):
    MSE = ((true - prediction) ** 2).mean()
    RMSE = np.sqrt(MSE)
    return MSE, RMSE

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
      real<lower=0.05, upper=0.2> eta;
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

    # All data is passed as kW quantities (out_power is the data to fit to)
    dat = {'n_obs': obs_power.shape[0], 
           'power': obs_power, 
           'I_obs': obs_irradiance,
           'area': peak_area, 
           'efficiency': eta,
           'n_forecast': for_irradiance.shape[0],
           'I_forecast': for_irradiance}

    sm = pystan.StanModel(model_code=model)

    return sm.sampling(data=dat, iter=30000, n_jobs=-1, control={'adapt_delta':0.95})


if __name__ == '__main__':

    # Train-test period definition
    train_start = [datetime(2020, 3, 27, tzinfo=timezone.utc) + timedelta(days=d) for d in range(1)]
    train_end = [datetime(2020, 4, 3, tzinfo=timezone.utc) + timedelta(days=d) for d in range(1)]

    test_start = [datetime(2020, 4, 3, tzinfo=timezone.utc) + timedelta(days=d) for d in range(1)]
    test_end = [datetime(2020, 4, 4, tzinfo=timezone.utc) + timedelta(days=d) for d in range(1)]

    # Read panels data and extract data about Liège (province)
    solar_stats = pd.read_csv('../resources/csv/solar_power_stats.csv', sep=';', thousands=".", 
                              index_col=0)
    kVA = solar_stats['kVA']
    ETA = 0.15 

    # Squared meters of PV panels per kWPeak in Belgium
    # Source: https://energieplus-lesite.be/concevoir/photovoltaique/
    #                 predimensionner-l-installation/
    AREA_PEAK = 7

    # Average area of panels in the province of Liège
    A = (AREA_PEAK * kVA).sum()

    # Data is in UTC. Forecast data starts on April 2nd (10:15)
    irradiance_observed = pd.read_csv('../cache/csv/observed_irradiance.csv', index_col=0)
    irradiance_observed.index = pd.to_datetime(irradiance_observed.index)

    irradiance_forecast = pd.read_csv('../cache/csv/forecast_irradiance.csv', index_col=0)
    irradiance_forecast.index = pd.to_datetime(irradiance_forecast.index)

    # WARNING: The corresponding Elia measures should have been cached beforehand (by calling 'cache_measures_between')
    # from 'power.py'.
    elia = get_cached_measures('solar')
    elia = elia.groupby(by=elia.index // 1800).mean()
    elia.index *= 1800
    elia.index = np.apply_along_axis(lambda l: [datetime.fromtimestamp(x, tz=timezone.utc) for x in l], 0, elia.index)

    elia['corrected'] = elia['corrected'].astype(float)
    elia['most_recent'] = elia['most_recent'].astype(float)

    for START, END, START_FOR, END_FOR in zip(train_start, train_end, test_start, test_end):

        # Slice appropriate period
        obs_irradiance = irradiance_observed[START:END]
        for_irradiance = irradiance_forecast[START_FOR:END_FOR]
        
        train_elia = elia[(elia.index >= START) & (elia.index < END)]
        if train_elia.isnull().values.any():
            train_elia.fillna(method="ffill", inplace=True)

        test_elia = elia[(elia.index >= START_FOR) & (elia.index < END_FOR)]
        if test_elia.isnull().values.any():
            test_elia.fillna(method="ffill", inplace=True)
        
        # Compute prior output power
        naive_obs_power = ETA * obs_irradiance * A / 1e6
        naive_for_power = ETA * for_irradiance * A / 1e6
        
        # Fit PyStan model on Elia's measures
        fit = pystan_model(train_elia['corrected'] * 1e3, AREA_PEAK, ETA, 
                           obs_irradiance['GHI'] / 1e3, for_irradiance['GHI'] / 1e3)
        post_samples = fit.extract()

        # Compare predicted distribution to Elia
        forecast_power = post_samples['for_power'] / 1e3
        mu_for = np.mean(forecast_power, axis=0)
        std_for = 3 * np.std(forecast_power, axis=0)
        
        # Performance assessment (remove nigth predictions) for naive and posterior models
        elia_metric = test_elia[test_elia['corrected'] > 0]
        naive_metric = naive_for_power[(test_elia['corrected'] > 0).values]
        mu_for_metric = mu_for[test_elia['corrected'] > 0]
        
        prior_MSE, prior_RMSE = compute_metrics(elia_metric['corrected'].values, naive_metric['GHI'].values)
        post_MSE, post_RMSE = compute_metrics(elia_metric['corrected'].values, mu_for_metric)

        # Performance assessment (remove nigth predictions) for Elia's predictions
        elia_pred = test_elia['most_recent'].astype('float64')
        elia_pred_metric = elia_pred[test_elia['corrected'] > 0]
        elia_MSE, elia_RMSE = compute_metrics(elia_metric['corrected'], elia_pred_metric)
        
        # Comparison with baseline models
        ## 1st baseline model: predict on day D+1 the measured production of day D
        baseline_1 = elia[(elia.index >= START_FOR - timedelta(days=1)) & (elia.index < START_FOR)]
        baseline_1.index += timedelta(days=1)
        baseline_1.drop(baseline_1.index[~baseline_1.index.isin(test_elia.index)], inplace=True)

        base_1_MSE, base_1_RMSE = compute_metrics(elia_metric['corrected'], baseline_1[test_elia['corrected'] > 0]['corrected'])

        ## 2nd baseline model: predict on day D+1 the average of the training days
        baseline_2 = train_elia
        baseline_2['Hour'] = baseline_2.index.time
        baseline_2 = baseline_2.groupby('Hour').mean()
        baseline_2.drop(baseline_2.index[~baseline_2.index.isin(test_elia.index.time)], inplace=True)
        baseline_2.index = test_elia.index
        
        base_2_MSE, base_2_RMSE = compute_metrics(elia_metric['corrected'], baseline_2[test_elia['corrected'] > 0]['corrected'])
        
        # Write results.
        # WARNING: If the file already exists, it will append the computed metrics to that file.
        metrics = np.array([[prior_MSE, prior_RMSE, post_MSE, post_RMSE, elia_MSE, elia_RMSE, base_1_MSE, base_1_RMSE, base_2_MSE, base_2_RMSE]])
        with open('../products/txt/results_provincial_april.txt','a') as f:
            np.savetxt(f, metrics, delimiter=' ')

        # Compare all models
        forecast_time_series = pd.date_range(start=for_irradiance.index[0], end=for_irradiance.index[-1], freq='30T', closed=None)
        fig, ax = plt.subplots()

        ax.fill_between(forecast_time_series, mu_for - std_for, mu_for + std_for, color="b", alpha=0.1)
        ax.plot(forecast_time_series, naive_for_power, color="k")
        ax.plot(forecast_time_series, mu_for, color="b")
        ax.plot(test_elia.index, test_elia['most_recent'], color="y")
        ax.plot(test_elia.index, test_elia['corrected'], color="orange")
        ax.plot(baseline_1.index, baseline_1['corrected'], color="r")
        ax.plot(baseline_2.index, baseline_2['corrected'], color="g")

        ax.set_xlabel('Time')
        ax.set_ylabel('Output power (MW)')
        fig.autofmt_xdate()
        ax.legend(['Naive', 'Posterior $\pm$ 3std', 'Elia forecast', 'Elia measurements', 'D-1', 'Avg(D-7)'])

        ax.grid()
        fig.tight_layout()
        fig.savefig('../products/pdf/solar_provincial (START_FOR {}).pdf'.format(START_FOR.strftime("%d-%m-%Y")), transparent=True)

        print("Predicted for {}".format(START_FOR.strftime("%d-%m-%Y")))

