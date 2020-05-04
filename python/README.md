## Solar Power Forecasting

Two scripts can be executed to get an estimation of the solar power that will be produced tomorrow in the province of Liège:

- `solar_panelwise.py`: **estimate tomorrow's production** by computing a power curve for each panel installation detected by our panel enumerator.
- `solar_provincial.py`: **estimate tomorrow's production** by computing a power curve directly for the whole province, using Elia's production measurements for the past 7 days to build a posterior predictive model.

Both caches present in the computational flow can be updated using the following scripts:

|                Cache | Script to use          |
| -------------------: | ---------------------- |
| SOLAR POWER MEASURES | `cache/power.py`       |
|  IRRADIANCE MEASURES | `cache/cache_solar.py` |
| IRRADIANCE FORECASTS | `cache/cache_solar.py` |



## Wind Power Forecasting

The following scripts are available. The first us to forecast the wind power that will be produced today and tomorrow in the whole Wallonia, and the second assess the final model built.
 - `wind_forecast.py [-h] [--display] {qxt,qgb}`: **forecast the wind power production** for today and tomorrow, using live weather prediction.
 - `wind_eval.py [-h] [--forecast] {qxt,qgb}`: **evaluate the current cached model** on the validation set, or the *weather forecast* test set if the `--forecast` option is used.

 The different caches of the computation flow can be updated using the following scripts:

| Cache | Script to use |
| -----:|:------------- |
| FARMS LOCATIONS | `cache/wind_farms.py` |
| WIND POWER MEASURES | `cache/power.py` |
| WIND WEATHER MEASURES | `cache/wind_weather.py [--start START] [--end END]` |
| WIND WEATHER FORECAST | `cache/wind_weather.py --forecast` |
| LEARNING SET | `cache/wind_ls.py` |
| TEST SET | `cache/wind_ls.py --test` |
| MODEL | `cache/wind_model.py [--train-set-only] {qxt,qgb}` |

The wind power forecast computation flow is display hereafter.

![Wind Power Forecast Computation Flow](../resources/png/wind_computation_flow.png)