import os
import pandas as pd
import numpy as np
import datetime as dt
import json
import ctypes
from functools import partial
import multiprocessing as mp
from category_encoders.target_encoder import TargetEncoder
from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from itertools import chain

from global_utils import file_utils
from . import utils
lock = mp.Lock()


def create_features(base_path: str, config, **kwargs):
    """
    Main Entry point to create features
    
    Args:

        base_path: path of the input data 
        config: global config
        **kwargs:Keyword Arguments

    Returns:

    """

    developer = FeatureDeveloper(base_path, config)
    developer.develop_features()

def create_group_features(config, feature_groups, *args, **kwargs):
    """
    Group Feature creation

    Args:
        config:
        feature_groups:
        *args:
        **kwargs:

    """

    group_developer = GroupFeatureDeveloper(config, feature_groups)
    return group_developer.create_group_features(*args, **kwargs)

@dataclass
class FeatureDeveloper:
    base_path: str
    config: Any

    def develop_features(self):
        data, future_data = self.get_input_data(self.base_path, self.config)
        common_features = self.get_common_features(data, self.base_path, self.config)
        group_args = self.create_group_args(data)
        
        features_path = f"{self.base_path}feature_engg" 
        path_to_write = f'{features_path}'
        if os.path.exists(path_to_write):
            os.system("rm -rf " + path_to_write)
        file_utils.makedirs(path_to_write, exist_ok = True)
        
        self.process_groups(group_args, future_data, common_features, features_path)

    def get_input_data(self, base_path: str, config):
        """

        Load clean data for feature engineering
        Args:
            base_path: base path of the customer data
            config: global config

        Returns:
            tuple(historic data, future data). Future data is None if it is not available

        """

        base_data_path = self.get_data_prep_path(base_path, config)
        data_path = f'{base_data_path}/{config.data_preparation.dump_data.input_data}.csv'
        columns = list({config.data.timestamp_column, *config.data.ts_id_columns,
                        *config.data.regressor_columns.numeric, *config.data.regressor_columns.categorical, 
                        config.data.target_column})
        data = pd.read_csv(file_utils.get_file_object(data_path), parse_dates=[config.data.timestamp_column],
                        usecols=columns)
        future_data = None
        if config.data_preparation.dump_data.get('future_data'):
            data_path = f'{base_data_path}/{config.data_preparation.dump_data.future_data}.csv'
            columns = [config.data.timestamp_column, *config.data.ts_id_columns,
            *config.data.regressor_columns]
            future_data = pd.read_csv(file_utils.get_file_object(data_path), parse_dates=[config.data.timestamp_column],
                                    usecols=columns)
        return data, future_data

    def get_data_prep_path(self, base_path: str, config):
        """
        
        Generate the path of the latest version of prepared data
        Args:
            base_path: base path of the customer data
            config: global config 

        Returns:

        """

        data_prep_path = f'{base_path}data_prepared'
        return f'{data_prep_path}'

    def get_common_features(self, data: pd.DataFrame, base_path: str, config):
        """

        Generate features common to all the groups
        Args:
            data: prepared data
            base_path: base path of the customer data
            config: global config json

        Returns:
            List of feature groups based on feature_types defined in the config file
        """

        features_list = []
        datetime_index = data.set_index(config.data.timestamp_column).resample(config.data.frequency).count().index
        datetime_index = self.extend_dt_index(datetime_index)
        self.feature_groups = dict(regressor = config.data.regressor_columns.numeric, 
                                   temporal = config.data.temporal_columns)
        if 'datetime' in config.feature_engg.feature_types:
            features_list.append(self.create_datetime_features(datetime_index))
            self.feature_groups['datetime'] = features_list[-1].columns.tolist()[:1]
        if 'sinusoidal' in config.feature_engg.feature_types:
            features_list.append(self.create_sinusoidal_features(datetime_index))
            self.feature_groups['sinusoidal'] = features_list[-1].columns.tolist()
        return pd.DataFrame(index=datetime_index).join(features_list)

    def extend_dt_index(self, index):
        """
        
        Extend the Date to future
        Args:
            index:
        
        """
        steps = self.config.data.forecast_horizon * index.freq
        return pd.date_range(index.min(), index.max() + steps, freq=index.freq)

    def create_datetime_features(self, datetime_index):
        """
        
        Generate datetime related features
        Args:
            datetime_index: pandas series containing timestamps

        Returns:
            Dataframe containing datetime features

        """

        freq = self.config.data.frequency
        datetime_features = datetime_index.to_series().rank().to_frame('sequence')
        datetime_features['year'] = datetime_index.year.astype('category')
        if freq[0] == 'Y':
            return datetime_features

        datetime_features['quarter_of_year'] = datetime_index.quarter.astype('category')
        if freq[0] == 'Q':
            return datetime_features

        datetime_features['month_of_year'] = datetime_index.month.astype('category')
        datetime_features['month_of_quarter'] = ((datetime_index.month - 1) % 3 + 1).astype('category')
        if freq[0] == 'M':
            return datetime_features

        datetime_features['week_of_year'] = datetime_index.week.astype('category')
        datetime_features['week_of_quarter'] = ((datetime_index.week - 1) % 13 + 1).astype('category')
        datetime_features['week_of_month'] = (datetime_index.week - datetime_index.map(
            lambda x: x.replace(day=1)).week + 1).astype('category')
        if freq[0] == 'W':
            return datetime_features

        datetime_features['day_of_month'] = datetime_index.day.astype('category')
        datetime_features['day_of_week'] = datetime_index.dayofweek.astype('category')
        if freq[-1] == 'D':
            return datetime_features

        datetime_features['hour_of_day'] = datetime_index.hour.astype('category')
        return datetime_features


    def create_sinusoidal_features(self, datetime_index):
        """
        Generate sinusoidal features
        Args:
            datetime_index: pandas series containing timestamps
            freq: frequency of the data
            freq_max_count: maximum count of frequencies to generate sinusoidal features

        Returns:
            Dataframe containing sinusoidal features
        """
        seq = datetime_index.to_series().rank()
        period = utils.get_period(self.config.data.frequency)
        freq_max_count = self.config.feature_engg.sinusoidal_freq_max_count

        sinusoidal_features = pd.DataFrame(index=datetime_index)
        for step in range(1, min(period, freq_max_count)):
            sinusoidal_features[f'sin_{step}'] = np.sin(2 * np.pi * step * seq / period)
            sinusoidal_features[f'cos_{step}'] = np.cos(2 * np.pi * step * seq / period)
        return sinusoidal_features

    def create_group_args(self, data):
        """

        Group Args creation
        Args:
            data:

        """

        args = []
        if self.config.feature_engg.get('groups'):
            for group_config in self.config.feature_engg.groups:
                if group_config.group_id_columns:
                    grouped_data = data.groupby(group_config.group_id_columns)
                    args.extend((group_config.group_id_columns, group_id, data) for group_id, data in grouped_data)
                else:
                    args.append(((), (), data))
        else:
            args.append(((), (), data))
        return args

    def process_groups(self, args, future_data, common_features, features_path):
        if len(args) > 1:
            n_cpus = self.config.get('n_cpus')
            if not n_cpus:
                n_cpus = .8
            if isinstance(n_cpus, float):
                n_cpus = int(mp.cpu_count() * n_cpus)
            pool = mp.Pool(n_cpus)

            pool.map(partial(create_group_features, self.config, self.feature_groups, 
            future_data = future_data, common_features = common_features, features_path = features_path), args)
            pool.close()
            pool.join()
        else:
            create_group_features(self.config, self.feature_groups, args[0],
            future_data = future_data, common_features = common_features, features_path = features_path)

@dataclass
class GroupFeatureDeveloper:
    config: Any
    feature_groups: str

    def create_group_features(self, group, future_data, common_features, features_path):
        """
        Generate group specific features
        Args:
            group: tuple(group_id, group_data)
            future_data:
            common_features: list containing common feature groups
            features_path: path to store generated features
            config: global config json

        Returns:

        """
        group_id_columns, group_ids, data = group
        self.level_name = '_'.join(group_id_columns or ('all',))
        self.features_path = features_path
        if not isinstance(group_ids, tuple):
            group_ids = group_ids,

        data = self.make_base_data(data, common_features)
        future_data = self.make_future_data(data, common_features, future_data,group_id_columns,  group_ids)
        future_data = utils.fillna_future_data(data, future_data, self.config)

        if 'lagged' in self.config.feature_engg.feature_types:
            lagged_features = self.create_lagged_features(data, future_data)
            if 'lagged' not in self.feature_groups:
                self.feature_groups['lagged'] = lagged_features.columns.tolist()
            data = data.join(lagged_features)
            future_data = future_data.join(lagged_features)
        if 'offset' in self.config.feature_engg.feature_types:
            offset_features = self.create_offset_features(data, future_data, target = True)
            self.feature_groups['offset'] = offset_features.columns.tolist()
            data = data.join(offset_features)
            future_data = future_data.join(offset_features)
        if self.config.data.temporal_columns:
            if 'relative' in self.config.feature_engg.feature_types:
                relative_features = self.create_relative_features(data, future_data)
                self.feature_groups['relative'] = relative_features.columns.tolist()
                data = data.join(relative_features)
                future_data = future_data.join(relative_features)
        if self.config.data.temporal_counter:
            if 'counter' in self.config.feature_engg.feature_types:
                temporal_counter_features = self.create_temporal_counter_features(data, future_data)
                self.feature_groups['counter'] = temporal_counter_features.columns.tolist()
                data = data.join(temporal_counter_features)
                future_data = future_data.join(temporal_counter_features)
        return self.create_data_splits(data, future_data)

    def make_base_data(self, data, common_features):
        """
        
        Transform input data based on the config parameters
        Args:
            data: input data
            common_features: Datetime and Sin/Cos features

        Returns:
            Dataframe containing transformed data
        """

        return (data.groupby(self.config.data.ts_id_columns)
                .apply(partial(self.make_ts, common_features=common_features))
                .join(common_features, on = self.config.data.timestamp_column))

    def make_ts(self, data, common_features, is_future=False):
        """

        Create the Time Series data
        Args:
            data:
            common_features:
            is_future:

        """

        if is_future:
            datetime_index = common_features.index[-self.config.data.forecast_horizon:]
        else:
            start_ind = (common_features.index >= data.loc[:, self.config.data.timestamp_column].min()).argmax()
            datetime_index = common_features.index[start_ind: -self.config.data.forecast_horizon]

        data = (self.fillna(data.drop(columns=list({*self.config.data.ts_id_columns}))
                .set_index(self.config.data.timestamp_column))
                .resample(self.config.data.frequency)
                .aggregate(self.config.data.aggregation_method)
                .reindex(datetime_index)
                .rename_axis(self.config.data.timestamp_column))

        if is_future:
            data[self.config.data.target_column] = 0
        return self.fillna(data)


    def fillna(self, data):
        """
        Fillna methods
        Args:
            data:
        
        """

        data.fillna(**self.config.data.fillna_method, inplace = True)
        data = data.fillna(method = 'ffill').fillna(method = 'bfill')
        numeric_regressors = self.config.data.regressor_columns.numeric
        categorical_regressors = self.config.data.regressor_columns.categorical
        data = data.fillna(value = dict([*zip(numeric_regressors, [0]*len(numeric_regressors)),
                                        *zip(categorical_regressors, ['UNKNOWN']*len(categorical_regressors))]))
        return data


    def make_future_data(self, data, common_features, future_data,group_id_columns, group_ids):
        """
        Transform future data based on the config parameters.
        If future data is not available, create a placeholder
        Args:
            data: input data
            common_features:
            future_data:
            group_id_columns: 
            group_ids:

        Returns:
            Dataframe containing transformed future data
        """

        if future_data is not None:
            mask = pd.Series(True, index=future_data.index)
            for k, v in zip(group_id_columns, group_ids):
                mask = mask & (future_data[k] == v)
            return (future_data.loc[mask]
                    .groupby(self.config.data.ts_id_columns)
                    .apply(partial(self.make_ts, common_features = common_features, is_future=True ))
                    .join(common_features, on = self.config.data.timestamp_column))
        index = data.index.droplevel(self.config.data.timestamp_column).unique()
        if len(self.config.data.ts_id_columns) > 1:
            future_index = pd.MultiIndex.from_tuples(
                [(*i, ts) for i in index for ts in common_features.index[-self.config.data.forecast_horizon:]],
                names = [*index.names, self.config.data.timestamp_column])
        else:
            future_index = pd.MultiIndex.from_tuples(
                [(i, ts) for i in index for ts in common_features.index[-self.config.data.forecast_horizon:]],
                names = [index.names[0], self.config.data.timestamp_column])
        future_data = (pd.DataFrame(index=future_index, columns=data.columns.difference(common_features.columns))
                .fillna(**self.config.data.fillna_method)
                .join(common_features, on = self.config.data.timestamp_column))
        future_data[self.config.data.target_column] = future_data[self.config.data.target_column].fillna(0).astype(float)
        return future_data

    def create_lagged_features(self, data, future_data):
        """
        Create lagged features
        Args:
            data:
            future_data:

        """

        lag_columns = [*self.config.data.temporal_columns, self.config.data.target_column]
        return (pd.concat([data, future_data])[lag_columns]
                        .groupby(level=self.config.data.ts_id_columns)
                        .apply(lambda x: self.create_ts_lagged_features(x)))

    def create_ts_lagged_features(self, data):
        """
        Create lagged features
        Args:
            data:

        Returns:
            Dataframe containing lagged features
        """

        lag_features = []
        lags = self.config.feature_engg.lag_params.get('values', [])
        if 'range' in self.config.feature_engg.lag_params:
            args = self.config.feature_engg.lag_params.range
            lags = [*lags, *range(*args)]
        assert 0 not in lags
        for lag in sorted(lags):
            lag_features.append(data.shift(lag)
                            .fillna(method = 'bfill')
                            .fillna(method = 'ffill')
                            .fillna(0)
                            .rename(columns=lambda x: f'{x}_{lag}'))
        return pd.concat(lag_features, axis=1) 

    def create_offset_features(self, data, future_data, target = False):
        """
        Create offset features
        Args:
            data:
            future_data:
            target:

        Returns:
            Dataframe containing offset features
        """

        offset_period = int(utils.get_period(self.config.data.frequency))
        if target:
            data = pd.concat([data, future_data])[[self.config.data.target_column]]
            offset_features = data.groupby(self.config.data.ts_id_columns).apply(partial(
                self.generate_offset, self.config.data.forecast_horizon, offset_period))
            del offset_features[self.config.data.target_column]
        return offset_features

    def generate_offset(self, lag_shift, offset_period, data):
        """

        Args:
            lag_shift:
            offset_period:
            data:

        Returns:

        """

        offset_params = self.config.feature_engg.offset_params
        data = utils.offset_functions(data, lag_shift, offset_period, offset_params)
        return self.fillna(data)
    
    def create_relative_features(self, data, future_data):
        """
        Create relative features
        Args:
            data:
            future_data:

        Returns:
            Dataframe containing relative features
        """

        relative_period = int(utils.get_period(self.config.data.frequency))
        relative_feature_cols =  self.config.data.get('relative_columns', self.config.data.temporal_columns)
        data = pd.concat([data, future_data])[relative_feature_cols]
        relative_features = (data.groupby(self.config.data.ts_id_columns)
                            .apply(partial(self.generate_relative, relative_feature_cols, relative_period)))
        return relative_features
    
    def generate_relative(self, relative_feature_cols, relative_period, data):
        """

        Args:
            relative_period:
            data:

        Returns:

        """

        relative_params = self.config.feature_engg.relative_params
        for feature in relative_feature_cols:
            data = utils.relative_functions(data, relative_period, relative_params, feature)
            del data[feature]
        return self.fillna(data)

    def create_temporal_counter_features(self, data, future_data):
        """
        Create temporal_counter features
        Args:
            data:
            future_data:

        Returns:
            Dataframe containing temporal_counter features
        """

        temporal_counter_cols =  self.config.data.temporal_counter
        data = pd.concat([data, future_data])[temporal_counter_cols]
        temporal_counter_features = (data.groupby(self.config.data.ts_id_columns)
                            .apply(partial(self.generate_temporal_counter, temporal_counter_cols)))
        return temporal_counter_features

    def generate_temporal_counter(self, temporal_counter_cols, data):
        """

        Args:
            temporal_counter_features:
            data:

        Returns:

        """

        for feature in temporal_counter_cols:
            data[feature + '_timesince'] = data.groupby((data[feature] != data[feature].shift()).cumsum()).cumcount() + 1
            data[feature + '_timesince'] = data[feature + '_timesince'].fillna(0)
            del data[feature]
        return self.fillna(data)

    def create_map(self, data):
        """
        Create lagged features
        Args:
            data:

        Returns:
            Dataframe containing lagged features
        """

        lag_features = []
        for lag in range(1, self.config.feature_engg.lag_count + 1):
            lag_features.append(data.shift(lag)
                                .fillna(method = 'bfill')
                                .fillna(method = 'ffill')
                                .rename(columns=lambda x: f'{x}_{lag}'))
        return pd.concat(lag_features, axis=1) 
    
    def create_data_splits(self, data, future_data):
        """
        Creates the different splits including train, val and test for following data -
        1. Future
        2. Virtual Future
        3. Valround1
        4. Valround2
        ..
        k. Valroundx
        
        Args:
            data:
            future_data:
        
        """

        ts_values = data.index.get_level_values(-1)
        ts_levels = data.index.levels[-1]
        forecast_horizon = self.config.data.forecast_horizon
        data['map'] = data.index.droplevel(-1).map(lambda x: '_'.join(map(str, x)))
        future_data['map'] = future_data.index.droplevel(-1).map(lambda x: '_'.join(map(str, x)))
        cat_cols = data.select_dtypes(exclude='number').columns.tolist()
        
        data_split_dict = {}

        #create future pipeline

        if len(ts_levels) <= 2 * forecast_horizon:
            return data_split_dict

        #round0
        train_timestamps = ts_levels[: -2 * forecast_horizon]
        val_timestamps = ts_levels[-2 * forecast_horizon: -forecast_horizon]
        test_timestamps = ts_levels[-forecast_horizon:]
        data_splits = [data.loc[ts_values.isin(train_timestamps)],
                        data.loc[ts_values.isin(val_timestamps)],
                        data.loc[ts_values.isin(test_timestamps)],
                        future_data]
        self.create_split_features(data_splits, cat_cols, 'future', 0)

        #round1 - roundn for future pipeline
        if self.config.feature_engg.valround == True:
            val_rounds = self.config.feature_engg.max_val_rounds
            stride = self.config.feature_engg.stride
            for val_round in range(val_rounds):
                val_round = val_round + 1
                train_timestamps = ts_levels[:-val_round * stride - 2 * forecast_horizon]
                if not len(train_timestamps):
                    break
                val_timestamps = ts_levels[-val_round * stride - 2 * forecast_horizon:
                                        -val_round * stride - forecast_horizon]
                test_timestamps = ts_levels[-val_round * stride - forecast_horizon:
                                            -val_round * stride]
                future_timestamps = ts_levels[-forecast_horizon:]
                data_splits = [data.loc[ts_values.isin(train_timestamps)],
                            data.loc[ts_values.isin(val_timestamps)],
                            data.loc[ts_values.isin(test_timestamps)],
                            data.loc[ts_values.isin(future_timestamps)]]
                self.create_split_features(data_splits, cat_cols, 'future', val_round)

        #create virtual future pipeline

        if len(ts_levels) <= 3 * forecast_horizon:
            return data_split_dict

        #round0
        train_timestamps = ts_levels[: -3 * forecast_horizon]
        val_timestamps = ts_levels[-3 * forecast_horizon: -2 * forecast_horizon]
        test_timestamps = ts_levels[- 2 * forecast_horizon: - forecast_horizon]
        future_timestamps = ts_levels[-forecast_horizon:]
        data_splits = [data.loc[ts_values.isin(train_timestamps)],
                        data.loc[ts_values.isin(val_timestamps)],
                        data.loc[ts_values.isin(test_timestamps)],
                        data.loc[ts_values.isin(future_timestamps)]]
        self.create_split_features(data_splits, cat_cols, 'virtual_future', 0)
        
        
        #round1 - roundn for virtual future pipeline
        if self.config.feature_engg.valround == True:
            val_rounds = self.config.feature_engg.max_val_rounds
            stride = self.config.feature_engg.stride
            for val_round in range(val_rounds):
                val_round = val_round + 1
                train_timestamps = ts_levels[:-val_round * stride - 3 * forecast_horizon]
                if not len(train_timestamps):
                    break
                val_timestamps = ts_levels[-val_round * stride - 3 * forecast_horizon:
                                        -val_round * stride - 2 * forecast_horizon]
                test_timestamps = ts_levels[-val_round * stride - 2*forecast_horizon:
                                            -val_round * stride - forecast_horizon]
                future_timestamps = ts_levels[-forecast_horizon:]
                data_splits = [data.loc[ts_values.isin(train_timestamps)],
                            data.loc[ts_values.isin(val_timestamps)],
                            data.loc[ts_values.isin(test_timestamps)],
                            data.loc[ts_values.isin(future_timestamps)]]
                self.create_split_features(data_splits, cat_cols, 'virtual_future', val_round)

        return data_split_dict

    def create_split_features(self, data_splits, cat_cols, future_type, val_round):
        """
        
        Generate features based on training data, concatenate common features and save them to files
        Args:
            data_splits: [train_data, val_data, test_data]
            cat_cols: list of categorical columns to be encoded

        Returns:

        """
        scaler = MinMaxScaler()

        cat_data = data_splits[0][cat_cols].reset_index(list(set(self.config.data.ts_id_columns)))
        cat_cols_present = cat_data.shape[1] > 0
        label_mapping = [
        {'col': col, 'mapping': dict(zip(cat_data[col].unique(), range(cat_data[col].nunique())))}
        for col in cat_data.columns]
        target_encoder = TargetEncoder(cols=cat_data.columns.tolist())
        label_encoder = OrdinalEncoder(cols=cat_data.columns.tolist(), mapping=label_mapping)
        label_encoded_cols = []

        for split, data in zip(['train', 'val', 'test', 'future'], data_splits):
            if cat_cols_present:
                split_features = []
                cat_data = data[cat_cols]
                for c in set(self.config.data.ts_id_columns):
                    cat_data.loc[:, c] = cat_data.index.get_level_values(c)
                if split == 'train':
                    _ = [cat_data[c].cat.remove_unused_categories(inplace=True)
                        for c in cat_data.select_dtypes(include='category').columns]
                    split_features.append(target_encoder.fit_transform(cat_data, data[self.config.data.target_column])
                                        .rename(columns='{}_target_encoded'.format))
                    split_features.append(label_encoder.fit_transform(cat_data)
                                      .rename(columns='{}_label_encoded'.format)
                                      + 2)
                    label_encoded_cols = split_features[-1].columns
                    if 'target_encoded' not in self.feature_groups:
                        self.feature_groups['target_encoded'] = split_features[0].columns.tolist()
                        self.feature_groups['label_encoded'] = label_encoded_cols.tolist()
                else:
                    split_features.append(target_encoder.transform(cat_data)
                                        .rename(columns='{}_target_encoded'.format))
                    split_features.append(label_encoder.transform(cat_data)
                                      .astype(int)
                                      .rename(columns='{}_label_encoded'.format)
                                      + 2)
                data = data.join(split_features)

            if split == 'train':
                numeric_cols = (data.select_dtypes(include='number')
                                .columns
                                .drop([self.config.data.target_column, *label_encoded_cols ])
                                .tolist())
                if numeric_cols:
                    data.loc[:, numeric_cols] = scaler.fit_transform(data.loc[:, numeric_cols])
            elif numeric_cols:
                data.loc[:, numeric_cols] = scaler.transform(data.loc[:, numeric_cols])

            if split == 'train':
                data_temp = data.reset_index()
                output = {'cols': list(data_temp.columns.values), 'dtypes': data.dtypes.apply(lambda x: x.name).to_dict(), 'feature_groups': dict(self.feature_groups)}
                output_path = f'{self.features_path}/output.json'
                with open(output_path, 'w') as f:
                    json.dump(output, f)

            path_to_write = f'{self.features_path}/{self.level_name}/{future_type}/round{val_round}'
            file_utils.makedirs(path_to_write, exist_ok = True)
            _filename = path_to_write + '/' + f'{split}.csv'
            with open(_filename, 'a') as f:
                data.to_csv(f, header=None)