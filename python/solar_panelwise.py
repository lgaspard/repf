import numpy as np
import pandas as pd
import tools.plot_setup
import matplotlib.pyplot as plt

import pvlib as pv
import pystan
import pytz

from datetime import datetime, timezone, timedelta, date, time
from cache.power import get_cached_measures, get_power_between
from cache.solar_weather import cache_irradiance_measures
from cache.solar_weather import cache_irradiance_forecasts


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
      real<lower=0, upper=0.2> eta;
      vector<lower=0>[n_forecast] forecast_power;  // True forecasts (not observed)
    }

    model {
      eta ~ normal(efficiency, 0.1);
      naive_for_power ~ normal(forecast_power, 10);
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


def solar_panelwise():

    # Datetime definitions
    tz = pytz.timezone('UCT')
    tomorrow = date.today() + timedelta(days=1)

    # Forecasting period
    START_FOR = datetime.combine(tomorrow, time())
    START_FOR = tz.localize(START_FOR)
    END_FOR = START_FOR + timedelta(days=1)

    # Training period
    START = START_FOR - timedelta(days=8)
    END = START_FOR - timedelta(days=1)

    # Panels hyperparameters (efficiency and tilt)
    ETA = 0.162
    BETA = np.deg2rad(10)

    # Retrieve cached irradiance data
    obs_irradiance, for_irradiance = get_cached_irradiance(
        START, END, START_FOR, END_FOR)

    # Check whether cached irradiance measures were up to date
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

    # Retrieve panels data
    panels = pd.read_csv('../resources/csv/liege_province.csv')

    latitudes = np.deg2rad(np.array(panels['lat']))
    longitudes = np.deg2rad(np.array(panels['lon']))
    azimuth = np.deg2rad(np.array(panels['azimuth']))
    areas = np.array(panels['area'])

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

    # Retrieve Elia's production data
    elia = get_power_between('solar', START, END_FOR)
    while elia is None:
        elia = get_power_between('solar', START, END_FOR)

    elia = elia.groupby(by=elia.index // 1800).mean()
    elia.index *= 1800
    elia.index = np.apply_along_axis(lambda l: [datetime.fromtimestamp(x, tz=timezone.utc) for x in l],
                                     0, elia.index)

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

    # Fit PyStan model on Elia's measures
    fit = pystan_model(naive_obs_power / 1e6,
                       train_elia['corrected'], naive_for_power / 1e6, ETA)
    post_samples = fit.extract()

    # Retrieve predicted power and compute mean distribution
    # post_samples['for_power'] is in MW
    forecast_power = post_samples['for_power']
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
    ax.legend(['Posterior forecast $\pm$ 3std', "Elia's forecast"])

    ax.grid()
    fig.tight_layout()
    fig.savefig('../products/pdf/solar_panelwise.pdf', transparent=True)

    return forecast_time_series, mu_for, mu_for - 3 * std_for, mu_for + 3 * std_for


if __name__ == '__main__':

    # Computes the mean distribution of the posterior predicted power, along
    # with +- 3std for tomorrow, using data computed by the photovoltaic panels
    # mapping. For the sake of convenience, we also return the forecast time
    # series.
    # To compute for another date, change 'tomorrow' assignment.
    solar_panelwise()
