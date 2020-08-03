import json
from global_utils import file_utils
import pandas as pd
import numpy as np


def get_period(freq):
        period = 1
        if freq[0] == 'Q':
                period = 4
        elif freq[0] == 'M':
                period = 12
        elif freq[0] == 'W':
                period = 52.18
        elif freq[-1] == 'D':
                period = 365.25 / int(freq[:-1] or 1)
        return period

def fillna_future_data(data, future_data, config):
        cols = data.columns
        data = pd.concat([data, future_data], axis = 0)
        data = data[cols]
        data = data.groupby(level= config.data.ts_id_columns).fillna(method = 'ffill').fillna(method = 'bfill')
        future_data = data.groupby(config.data.ts_id_columns).tail(config.data.forecast_horizon)
        return future_data

def offset_functions(data, lag_shift, offset_period, offset_params):
        for function in offset_params.function_types:
            for period in offset_params.period_types:
                if function == 'sum':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .sum()
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'mean':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .mean()
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'median':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .median()
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'min':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .min()
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'max':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .max()
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'kurtosis':
                        if offset_period//period > 4:
                                data[function + '_offset_' + str(offset_period//period)] = np.nan
                                data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                        .rolling(offset_period//period)
                                                                                        .kurt()
                                                                                        .fillna(method = 'bfill')
                                                                                        .fillna(0))
                if function == 'variance':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .var()
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'quantile_10':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .quantile(.1, interpolation='midpoint')
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'quantile_90':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .quantile(.9, interpolation='midpoint')
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
                if function == 'skewness':
                        data[function + '_offset_' + str(offset_period//period)] = np.nan
                        data[function + '_offset_' + str(offset_period//period)] = (data.shift(lag_shift)
                                                                                .rolling(offset_period//period)
                                                                                .skew()
                                                                                .fillna(method = 'bfill')
                                                                                .fillna(0))
        return data

def relative_functions(data, relative_period, relative_params, feature):
        for period in relative_params.relative_periods:
                data[feature + '_relative_' + str(relative_period//period)] = np.nan
                data[feature + '_relative_' + str(relative_period//period)] = (data.shift(1)
                                                                        .rolling(relative_period//period)
                                                                        .mean()
                                                                        .fillna(method = 'bfill')
                                                                        .fillna(0))
                data[feature + '_relative_' + str(relative_period//period)] = (data[feature]/data[feature + '_relative_' + str(relative_period//period)])
                data[feature + '_relative_' + str(relative_period//period)]  = data[feature + '_relative_' + str(relative_period//period)].replace([np.inf, -np.inf], 0).fillna(0)
        return data