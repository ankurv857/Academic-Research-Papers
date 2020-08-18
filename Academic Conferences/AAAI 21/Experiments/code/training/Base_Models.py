from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from training.Metrics import BCE
import copy
from copy import deepcopy
from itertools import chain

from . import utils
np.random.seed(0)
torch.random.manual_seed(0)


class BaseModel(ABC):
    _registry: Dict[str, Any] = {}

    def __init_subclass__(cls, custom_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if custom_name is None:
            custom_name = cls.__name__
        assert custom_name not in cls._registry, \
            f'Model with name {custom_name} is already registered: {cls._registry[custom_name]}!'
        cls._registry[custom_name] = cls

    @abstractmethod
    def fit(self, X, y, val_data=None, **fit_params):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @classmethod
    def from_name(cls, name):
        assert name in cls._registry, f"""Invalid model name '{name}'. Model name must be one of '{
            list(cls._registry.keys())}'"""
        return cls._registry[name]

    def cross_validate(self, cv_data, config, model_config, **fit_params):
        """
            Runs the cross validated model and generates metrics

            Args:
                cv_data:
                trials:
                fit_params:
        
        """

        self.config = config
        metrics_dict = {}
        for data_key in cv_data:
            if cv_data[data_key]:
                metric_list = []
                for _round, data_splits in enumerate(cv_data[data_key]):
                    train_split, val_split, test_split, future_split = data_splits
                    self.fit(*train_split, val_split, **fit_params)
                    metric_list.append(self.get_round_metrics(data_splits, _round, data_key, model_config))
                    
                model_metrics_dict, ts_metric_list = list(zip(*metric_list))
            else:
                model_metrics_dict, ts_metric_list = {}, []
            metrics_dict[data_key] =  dict(model_metrics = list(model_metrics_dict), ts_metrics = list(chain(*ts_metric_list)))
        return metrics_dict

    def get_round_metrics(self, data_splits: list, _round, data_key, model_config):
        """
        Compute the round metric at different levels including ts_level
        Args:
            data_splits:
            _round:
            data_key:

        """

        self.model_config = model_config
        model_metrics_dict = {}
        ts_metric_list = []
        for split, (X,y) in zip(('train', 'val', 'test', 'future'), data_splits):
            X, y = utils.seq_len_tail(X, y, self.config, self.model_config)
            pred = self.predict(X)
            if len(pred) != len(y):
                model_metrics_dict[split] = 0.0
                result_df = y.to_frame('actual')
                result_df['pred'] = y.to_frame('actual')
            else:
                model_metrics_dict[split] = BCE(y, self.predict(X))
                result_df = y.to_frame('actual')
                result_df['pred'] = pred
            ts_df = copy.deepcopy(result_df)
            if _round == 0 :
                ts_metric_list.append(dict(
                    _round = _round, data_split = split, ts_id = 'ts_id',
                    metric_value = BCE(ts_df.actual, ts_df.pred),
                    pred = [ts_df]))
            else:
                ts_metric_list.append(dict(
                    _round = _round, data_split = split, ts_id = 'ts_id',
                    metric_value = BCE(ts_df.actual, ts_df.pred),
                    pred = []))
        return model_metrics_dict, ts_metric_list

    def model_criteria(self, cv_data, model_config):
        """
            Applies model level criteria for model run qualification
            Args:
                train_split:
                model_config:

        """

        model_status = True
        X = cv_data['cv_data'][0][0][0]
        if (model_config.min_obj_size == True) & (len(X.index.get_level_values(self.config.data.timestamp_column).unique()) < self.config.data.forecast_horizon):
            model_status = False
        return model_status

    def _scale_y(self, y, fit=False):
        """
            scale the target
            Args:
                y:
                fit:

        """

        return y

    def _invert_y(self, y):
        """
            Invert scale the target
            Args:
                y:

        """
        
        return y
        
    def score(self, X, y, sample_weight=None):
        """

        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        return r2_score(y, self.predict(X), sample_weight)