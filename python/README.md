## Solar Power Forecasting

TDB

## Wind Power Forecasting

The interface is composed of:
 - `wind_eval.py [-h] [--forecast]`: **evaluate the current cached model** on the validation set, or the test set if the `--forecast` option is used.
 - `wind_forecast.py [-h] [--display]`: **forecast the wind power production** for today and tomorrow, using live weather prediction.

 The different caches of the computation flow can be updated using the following scripts:

| Cache | Script to use |
| -----:|:------------- |
| FARMS LOCATIONS | `cache/wind_farms.py` |
| WIND POWER MEASURES | `cache/power.py` |
| WIND WEATHER MEASURES | `cache/wind_weather.py [--start START] [--end END]` |
| WIND WEATHER FORECAST | `cache/wind_weather.py --forecast` |
| LEARNING SET | `cache/wind_ls.py` |
| TEST SET | `cache/wind_ls.py --test` |
| MODEL | `cache/wind_model.py [--train-set-only]` |

The wind power forecast computation flow is display hereafter.

![Wind Power Forecast Computation Flow](../resources/png/wind_computation_flow.png)