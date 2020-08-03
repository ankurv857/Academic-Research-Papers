from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import Any
import copy
from copy import deepcopy
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from . import utils

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
    
    def run_stacker(self, model_data, config, future_type, **fit_params):
        """
        Runs the stacker over the predicted dataset
        Args:
            model_data:
            fit_params:

        """
        
        for data_key in model_data:
            train_split, val_split, test_split, future_split = model_data[data_key]
            X = pd.concat([train_split[0], val_split[0], test_split[0]], axis = 0).fillna(0)
            y = pd.concat([train_split[1], val_split[1], test_split[1]], axis = 0)
            if len(X) != len(y):
                result_df = pd.DataFrame()
                return pd.DataFrame(result_df, columns = ['actual', 'pred'])
            else:
                self.fit(X,y, val_split, **fit_params)
                for split, (X,y) in zip(('train', 'val', 'test', 'future'), model_data[data_key]):
                    if split == 'future':
                        pred = self.predict(X)
                        result_df = pd.DataFrame(y).assign(pred = pred)
                        result_df.columns = ['actual','pred']
                return result_df

@dataclass
class K_best:
    model_data: str
    config: Any
    
    def model_compute(self, future_type):
        """
        Model K_best
        Args:
            future_type:

        """

        metrics = pd.DataFrame()
        for data_list_key in self.model_data.keys():
            for split, data in zip(['train', 'val', 'test', 'future'], self.model_data[data_list_key]):
                predictions_data = utils.generate_predictions(split, data, self.config)
                if split == 'future':
                    predictions = copy.deepcopy(predictions_data)
                metric = utils.metric_cal(split, predictions_data, future_type)
                metrics = pd.concat([metrics, metric], axis = 0)
        return predictions, metrics

class _ElasticNet(ElasticNet, BaseModel):

    def __init__(self, config, **kwargs):
        """
        Elastic Net model stacker
        Args:
            kwargs:
            config:

        """

        super().__init__(**kwargs)
        self.config = config
    
    def fit(self, X, y, val_data=None, **kwargs):
        """
        fit function 
        Args:
            X:
            y:
        
        """

        super().fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """
        Predict function
        Args:
            X:
        
        """

        return super().predict(X, **kwargs)

class _Lasso(Lasso, BaseModel):

    def __init__(self, config, **kwargs):
        """
        Elastic Net model stacker
        Args:
            kwargs:
            config:

        """

        super().__init__(**kwargs)
        self.config = config
    
    def fit(self, X, y, val_data=None, **kwargs):
        """
        fit function 
        Args:
            X:
            y:
        
        """

        super().fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """
        Predict function
        Args:
            X:
        
        """

        return super().predict(X, **kwargs)

class _Ridge(Ridge, BaseModel):

    def __init__(self, config, **kwargs):
        """
        Elastic Net model stacker
        Args:
            kwargs:
            config:

        """

        super().__init__(**kwargs)
        self.config = config
    
    def fit(self, X, y, val_data=None, **kwargs):
        """
        fit function 
        Args:
            X:
            y:
        
        """

        super().fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """
        Predict function
        Args:
            X:
        
        """

        return super().predict(X, **kwargs)
    