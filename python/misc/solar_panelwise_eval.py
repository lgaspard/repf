import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import tools.plot_setup
import matplotlib.pyplot as plt

import pystan
import pvlib as pv

from cache.power import get_cached_measures, get_power_between
from datetime import datetime, timedelta, timezone


def compute_metrics(true, prediction):
    MSE = ((true - prediction) ** 2).mean()
    RMSE = np.sqrt(MSE)
    return MSE, RMSE


def hour_angle(times, longitudes, equation_of_time):
    """
    Hour angle in local solar time. Zero at local solar noon.

    Parameters
    ----------
    times: pd.DatetimeIndex
        Corresponding timestamps, must be localized to the timezone for the
        "longitude".
    longitudes: numeric
        Longitude(s) in radians.
    equation_of_time: numeric
        Equation of time in minutes.

    Returns
    -------
    hour_angle: numeric
        Hour angle in local solar time in radians.

    References
    ----------
    pvlib.solarposition.hour_angle
    """

    # ===== COPY PASTED =====
    NS_PER_HR = 1.e9 * 3600.  # Nanoseconds per hour

    naive_times = times.tz_localize(None)  # Naive but still localized
    hrs_minus_tzs = 1 / NS_PER_HR * (
        2 * times.astype(np.int64) - times.normalize().astype(np.int64) -
        naive_times.astype(np.int64)
    )
    # =======================

    hrs_minus_tzs = np.asarray(hrs_minus_tzs, dtype=float)

    return np.radians(15. * (hrs_minus_tzs - 12.) +
                      equation_of_time / 4.).values.reshape(-1, 1) + \
        longitudes.reshape(1, -1)


def elevation(times, latitudes, longitudes):
    """
    Elevation angle of the sun in local solar time. Zero at horizon.

    Parameters
    ----------
    times: pd.DatetimeIndex
        Corresponding timestamps, must be localized or UTC will be assumed.
    latitudes: numeric
        Latitude(s) in radians.
    longitudes: numeric
        Longitude(s) in radians.

    Returns
    -------
    elevation: numeric
        Elevation angle in local solar time in radians.

    References
    ----------
    pvlib.solarposition.ephemeris
    """

    # ===== COPY PASTED =====
    Abber = 20 / 3600.

    # The SPA algorithm needs time to be expressed in terms of
    # decimal UTC hours of the day of the year.

    # If localized, convert to UTC. otherwise, assume UTC.
    try:
        time_utc = times.tz_convert('UTC')
    except TypeError:
        time_utc = times

    # Strip out the day of the year and calculate the decimal hour
    DayOfYear = time_utc.dayofyear
    DecHours = (time_utc.hour + time_utc.minute/60. + time_utc.second/3600. +
                time_utc.microsecond/3600.e6)

    # np.array needed for pandas > 0.20
    UnivDate = np.array(DayOfYear)
    UnivHr = np.array(DecHours)

    Yr = np.array(time_utc.year) - 1900
    YrBegin = 365 * Yr + np.floor((Yr - 1) / 4.) - 0.5

    Ezero = YrBegin + UnivDate
    T = Ezero / 36525.

    # Calculate Greenwich Mean Sidereal Time (GMST)
    GMST0 = 6 / 24. + 38 / 1440. + (
        45.836 + 8640184.542 * T + 0.0929 * T ** 2) / 86400.
    GMST0 = 360 * (GMST0 - np.floor(GMST0))
    GMSTi = np.mod(GMST0 + 360 * (1.0027379093 * UnivHr / 24.), 360)
    # =======================

    # Local apparent sidereal time
    LocAST = (360 + GMSTi.reshape(-1, 1) +
              np.degrees(longitudes).reshape(1, -1)) % 360

    # ===== COPY PASTED =====
    EpochDate = Ezero + UnivHr / 24.
    T1 = EpochDate / 36525.

    ObliquityR = np.radians(
        23.452294 - 0.0130125 * T1 - 1.64e-06 * T1 ** 2 + 5.03e-07 * T1 ** 3)
    MlPerigee = 281.22083 + 4.70684e-05 * EpochDate + 0.000453 * T1 ** 2 + (
        3e-06 * T1 ** 3)
    MeanAnom = np.mod((358.47583 + 0.985600267 * EpochDate - 0.00015 *
                       T1 ** 2 - 3e-06 * T1 ** 3), 360)
    Eccen = 0.01675104 - 4.18e-05 * T1 - 1.26e-07 * T1 ** 2
    EccenAnom = MeanAnom
    E = 0

    while np.max(abs(EccenAnom - E)) > 0.0001:
        E = EccenAnom
        EccenAnom = MeanAnom + np.degrees(Eccen)*np.sin(np.radians(E))

    TrueAnom = (
        2 * np.mod(np.degrees(np.arctan2(((1 + Eccen) / (1 - Eccen)) ** 0.5 *
                                         np.tan(np.radians(EccenAnom) / 2.), 1)), 360))
    EcLon = np.mod(MlPerigee + TrueAnom, 360) - Abber
    EcLonR = np.radians(EcLon)
    DecR = np.arcsin(np.sin(ObliquityR)*np.sin(EcLonR))

    RtAscen = np.degrees(np.arctan2(np.cos(ObliquityR)*np.sin(EcLonR),
                                    np.cos(EcLonR)))
    # =======================

    HrAngleR = np.radians(LocAST - RtAscen.reshape(-1, 1))
    DecR = DecR.reshape(-1, 1)
    LatR = latitudes.reshape(1, -1)

    return np.arcsin(
        np.cos(DecR) * np.cos(HrAngleR) * np.cos(LatR) + np.sin(DecR)
        * np.sin(LatR)
    )


def compute_solar_angles(time_series, latitudes, longitudes, azimuth, beta):
    """
    Compute the cosine of the incidence angle of sun rays for each latitude and
    longitude as well as the corresponding sine of the solar altitudes.

    Parameters
    ----------
    time_series: pd.DatetimeIndex
        The time series with which the solar angles are computed
    latitudes: array of float
        The latitudes of each panel in radians
    longitudes: array of float
        The longitudes of each panel in radians
    azimuth: array of float
        The surface azimuth of each panel in radians

    Return
    ------
    sin_sun_alt: array of shape [n_time_steps, n_panels]
        The sine of the solar altitudes of each panel, for all time steps
    cos_incidence: array of shape [n_time_steps, n_panels]
        The cosine of the incidence angle on each panel, for all time steps
    """

    # Compute declination angle (radians)
    dec = np.array(pv.solarposition.declination_cooper69(
        time_series.dayofyear)).reshape(-1, 1)

    # Compute equation of time
    eq_time = pv.solarposition.equation_of_time_pvcdrom(time_series.dayofyear)

    # Compute hour angle (radians) of each panel
    h_angle = hour_angle(time_series, longitudes, eq_time)

    # Compute sine of solar altitudes of each panel
    sin_sun_alt = np.sin(elevation(time_series, latitudes, longitudes))

    latitudes = latitudes.reshape(1, -1)
    azimuth = azimuth.reshape(1, -1)

    # Compute cosine of incidence angle of sun rays on each panel
    cos_incidence = (np.sin(dec) * np.sin(latitudes) * np.cos(beta)
                     + np.sin(dec) * np.cos(latitudes) * np.sin(beta)
                     * np.cos(azimuth) + np.cos(dec) * np.cos(latitudes)
                     * np.cos(beta) * np.cos(h_angle) - np.cos(dec)
                     * np.sin(latitudes) * np.sin(beta) * np.cos(azimuth)
                     * np.cos(h_angle) - np.cos(dec) * np.sin(beta)
                     * np.sin(azimuth) * np.sin(h_angle))

    return sin_sun_alt, cos_incidence


def compute_out_power(sin_sun_alt, cos_incidence, areas, irradiance):
    """
    Compute the total output power of all panels for the time series defined by
    the input solar angles.

    Parameters
    ----------
    sin_sun_alt: array of shape [n_time_steps, n_panels]
        The sine of the solar altitudes of each panel, for all time steps
    cos_incidence: array of shape [n_time_steps, n_panels]
        The cosine of the incidence angle on each panel, for all time steps
    areas: array of float
        The area of each panel

    Return
    ------
    out_power: array of float
        The total power over the time series
    """

    areas = areas.reshape(1, -1)
    irradiance = irradiance['GHI'].values.reshape(-1, 1)

    # If both measures are negative, no irradiance should be perceived.
    # Equivalent to set the cosine to zero.
    cos_incidence[(cos_incidence <= 0) & (sin_sun_alt <= 0)] = 0

    out_power = irradiance * areas * cos_incidence / sin_sun_alt

    # Negative output power means no rays on the panel
    out_power[out_power <= 0] = 0

    return out_power.sum(axis=1)


def pystan_model(naive_obs_power, true_power, naive_for_power, eta):
    model = """
    data {
      int<lower=0> n_obs; 
      vector[n_obs] true_power;
      vector[n_obs] naive_obs_power;
      
      real<lower=0> efficiency; 

      int<lower=0> n_forecast; 
      vector[n_forecast] naive_for_power; 
    }

    parameters {
      real<lower=0.05, upper=0.2> eta;
      
      vector<lower=0>[n_forecast] forecast_power;  // True forecasts (not observed)
    }

    model {
      eta ~ normal(efficiency, 0.1);
      forecast_power ~ normal(naive_for_power, 10);
      true_power ~ normal(eta * naive_obs_power, 100);
    }

    generated quantities {
        vector[n_forecast] for_power; // Forecast output power

        for (i in 1:n_forecast) {
            for_power[i] = normal_rng(eta * forecast_power[i], 10);
        }
    }
    """

    # All data is passed as MW quantities (power is the data to fit to)
    dat = {'n_obs': true_power.shape[0],
           'true_power': true_power,
           'naive_obs_power': naive_obs_power,
           'efficiency': eta,
           'n_forecast': naive_for_power.shape[0],
           'naive_for_power': naive_for_power}

    sm = pystan.StanModel(model_code=model)

    return sm.sampling(data=dat, iter=30000, n_jobs=-1, 
                       control={'adapt_delta': 0.95})


if __name__ == '__main__':

    # Train-test period definition
    train_start = [datetime(2020, 3, 27, tzinfo=timezone.utc) +
                   timedelta(days=d) for d in range(5)]
    train_end = [datetime(2020, 4, 3, tzinfo=timezone.utc) +
                 timedelta(days=d) for d in range(5)]
    test_start = [datetime(2020, 4, 3, tzinfo=timezone.utc) +
                  timedelta(days=d) for d in range(5)]
    test_end = [datetime(2020, 4, 4, tzinfo=timezone.utc) +
                timedelta(days=d) for d in range(5)]

    # Panels hyperparameters (efficiency and tilt)
    ETA = 0.162
    BETA = np.deg2rad(10)

    # Data is in UTC. Forecast data starts on April 2nd (10:15)
    irradiance_observed = pd.read_csv('../cache/csv/observed_irradiance.csv', 
                                      index_col=0)
    irradiance_observed.index = pd.to_datetime(irradiance_observed.index)

    irradiance_forecast = pd.read_csv('../cache/csv/forecast_irradiance.csv', 
                                      index_col=0)
    irradiance_forecast.index = pd.to_datetime(irradiance_forecast.index)

    # WARNING: The corresponding Elia measures should have been cached 
    # beforehand (by calling 'cache_measures_between') from 'cache/power.py'.
    elia = get_cached_measures('solar')
    elia = elia.groupby(by=elia.index // 1800).mean()
    elia.index *= 1800
    elia.index = np.apply_along_axis(lambda l: [datetime.fromtimestamp(x, tz=timezone.utc) for x in l], 
                                     0, elia.index)

    elia['corrected'] = elia['corrected'].astype(float)
    elia['most_recent'] = elia['most_recent'].astype(float)

    # Retrieve panels data
    panels = pd.read_csv('../resources/csv/liege_province.csv')

    latitudes = np.deg2rad(np.array(panels['lat']))
    longitudes = np.deg2rad(np.array(panels['lon']))
    azimuth = np.deg2rad(np.array(panels['azimuth']))
    areas = np.array(panels['area'])

    # Evaluation loop
    for START, END, START_FOR, END_FOR in zip(train_start, train_end, test_start, test_end):

        # Slice appropriate period
        obs_irradiance = irradiance_observed[START:END]
        for_irradiance = irradiance_forecast[START_FOR:END_FOR]

        # Divide Elia's production data into training data and forecast data 
        # and check for NaN
        train_elia = elia[(elia.index >= START) & (elia.index < END)].copy()
        if train_elia.isnull().values.any():
            train_elia.interpolate(inplace=True)

        test_elia = elia[(elia.index >= START_FOR) &
                         (elia.index < END_FOR)].copy()
        if test_elia.isnull().values.any():
            test_elia.interpolate(inplace=True)

        # Compute solar angles for the considered observed and forecast time series
        time_series = pd.date_range(start=obs_irradiance.index[0], 
                                    end=obs_irradiance.index[-1], 
                                    freq='30T', 
                                    closed=None)
        sin_sun_alt, cos_incidence = compute_solar_angles(time_series, latitudes,
                                                          longitudes, azimuth,
                                                          BETA)
        naive_obs_power = compute_out_power(sin_sun_alt, cos_incidence, areas,
                                            obs_irradiance)

        time_series = pd.date_range(start=for_irradiance.index[0], 
                                    end=for_irradiance.index[-1], 
                                    freq='30T', 
                                    closed=None)
        sin_sun_alt, cos_incidence = compute_solar_angles(time_series, latitudes,
                                                          longitudes, azimuth,
                                                          BETA)
        naive_for_power = compute_out_power(sin_sun_alt, cos_incidence, areas,
                                            for_irradiance)

        # Fit PyStan model on Elia's measures
        fit = pystan_model(naive_obs_power / 1e6,
                           train_elia['corrected'], naive_for_power / 1e6, ETA)
        post_samples = fit.extract()

        # Retrieve predicted power and compute mean distribution (Watts)
        forecast_power = post_samples['for_power']
        mu_for = np.mean(forecast_power, axis=0)
        std_for = 3 * np.std(forecast_power, axis=0)

        # Performance assessment (remove nigth predictions) for naive and 
        # posterior models
        elia_metric = test_elia[test_elia['corrected'] > 0]
        naive_metric = naive_for_power[(test_elia['corrected'] > 0).values] * ETA / 1e6
        mu_for_metric = mu_for[test_elia['corrected'] > 0]

        prior_MSE, prior_RMSE = compute_metrics(elia_metric['corrected'].values, 
                                                naive_metric)
        post_MSE, post_RMSE = compute_metrics(elia_metric['corrected'].values, 
                                              mu_for_metric)

        # Performance assessment (remove nigth predictions) for Elia's predictions
        elia_pred = test_elia['most_recent'].astype('float64')
        elia_pred_metric = elia_pred[test_elia['corrected'] > 0]
        elia_MSE, elia_RMSE = compute_metrics(elia_metric['corrected'], 
                                              elia_pred_metric)

        # Comparison with baseline models
        # 1st baseline model: predict on day D+1 the measured production of day D
        baseline_1 = elia[(elia.index >= START_FOR - timedelta(days=1)) & (elia.index < START_FOR)]
        baseline_1.index += timedelta(days=1)
        baseline_1.drop(baseline_1.index[~baseline_1.index.isin(test_elia.index)], inplace=True)

        base_1_MSE, base_1_RMSE = compute_metrics(elia_metric['corrected'], 
                                                  baseline_1[test_elia['corrected'] > 0]['corrected'])

        # 2nd baseline model: predict on day D+1 the average of the training days
        baseline_2 = train_elia
        baseline_2['Hour'] = baseline_2.index.time
        baseline_2 = baseline_2.groupby('Hour').mean()
        baseline_2.drop(baseline_2.index[~baseline_2.index.isin(test_elia.index.time)], inplace=True)
        baseline_2.index = test_elia.index

        base_2_MSE, base_2_RMSE = compute_metrics(elia_metric['corrected'], 
                                                  baseline_2[test_elia['corrected'] > 0]['corrected'])

        # Write results.
        # WARNING: If the file already exists, it will append the computed metrics to that file.
        metrics = np.array([[prior_MSE, prior_RMSE, post_MSE, post_RMSE, elia_MSE,
                             elia_RMSE, base_1_MSE, base_1_RMSE, base_2_MSE, base_2_RMSE]])
        with open('../products/txt/results_panelwise_april.txt', 'a') as f:
            np.savetxt(f, metrics, delimiter=' ')

        # Compare all models
        forecast_time_series = pd.date_range(start=for_irradiance.index[0], 
                                             end=for_irradiance.index[-1], 
                                             freq='30T', 
                                             closed=None)
        
        fig, ax = plt.subplots()

        ax.fill_between(forecast_time_series, mu_for - std_for,
                        mu_for + std_for, color="b", alpha=0.1)
        ax.plot(forecast_time_series, naive_for_power * ETA / 1e6, color="k")
        ax.plot(forecast_time_series, mu_for, color="b")
        ax.plot(test_elia.index, test_elia['most_recent'], color="y")
        ax.plot(test_elia.index, test_elia['corrected'], color="orange")
        ax.plot(baseline_1.index, baseline_1['corrected'], color="r")
        ax.plot(baseline_2.index, baseline_2['corrected'], color="g")

        ax.set_xlabel('Time')
        ax.set_ylabel('Output power (MW)')
        fig.autofmt_xdate()
        ax.legend(['Naive', 'Posterior $\pm$ 3std', 'Elia forecast',
                   'Elia measurements', 'D-1', 'Avg(D-7)'])

        ax.grid()
        fig.tight_layout()
        fig.savefig('../products/pdf/solar_panelwise (START_FOR {}).pdf'.format(
            START_FOR.strftime("%d-%m-%Y")), transparent=True)

        print("Predicted for {}".format(START_FOR.strftime("%d-%m-%Y")))
