import os
import pandas as pd
import numpy as np
import random
import xgboost as xgb
import itertools
import copy
from copy import deepcopy
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from itertools import chain
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from . import utils
from .Base_Models import BaseModel

class Xgboost(xgb.XGBClassifier, BaseModel):

    def __init__(self, config, feature_groups = None, dtypes = None, model_name = None , **kwargs):
        """

        Args:
            config:
            model_path:
            **kwargs:

        """

        super().__init__(**kwargs)
        self.config = config
        self.model_config = self.config.training.models.get(model_name, {})
        self.usecols, self.dtypes  = utils.get_usecols(feature_groups, dtypes, self.model_config.feature_groups, 
                                        self.model_config.exclude_cols, config)
        self.max_log_y = 1

    def fit(self, X, y, val_data=None, trial=None, val_round=None, **kwargs):
        """

        Args:
            X:
            y:
            val_data:
            trial:
            val_round:
            **kwargs:

        Returns:

        """
        self.cat_cols = X.select_dtypes(exclude='number').columns
        X[self.cat_cols] = X[self.cat_cols].astype(float).astype(int)
        if val_data:
            val_data[0][self.cat_cols] = val_data[0][self.cat_cols].astype(float).astype(int)
        if self.model_config.scale == True:
            y = self._scale_y(y, fit=True)
            super().fit(X, y, eval_set= [(val_data[0],self._scale_y(val_data[1]))] , 
            early_stopping_rounds= self.model_config.early_stopping_rounds, verbose = False, **kwargs)
        else:
            super().fit(X, y, eval_set= [(val_data[0],val_data[1])] , 
            early_stopping_rounds= self.model_config.early_stopping_rounds, verbose = False, **kwargs)

    def predict(self, X, **kwargs):
        """

        Args:
            X:
            **kwargs:

        Returns:

        """
        X[self.cat_cols] = X[self.cat_cols].astype(float).astype(int)
        pred = super().predict_proba(X, **kwargs)
        pred = pd.DataFrame(pred)
        # print('pred', pred) ; exit()
        pred.columns = ['pred0', 'pred1']
        return pred['pred1'].values

    def get_params(self, deep=True):
        params = super().get_params(deep)
        cp = copy.copy(self)
        cp.__class__ = xgb.XGBClassifier
        params.update(cp.__class__.get_params(cp, deep))
        return params

class RandomForest(RandomForestClassifier, BaseModel):

    def __init__(self, config, feature_groups = None, dtypes = None, model_name = None , **kwargs):
        """

        Args:
            config:
            model_path:
            **kwargs:

        """

        super().__init__(**kwargs)
        self.config = config
        self.model_config = self.config.training.models.get(model_name, {})
        self.usecols, self.dtypes  = utils.get_usecols(feature_groups, dtypes, self.model_config.feature_groups, 
                                        self.model_config.exclude_cols, config)
        self.max_log_y = 1

    def fit(self, X, y, val_data=None, trial=None, val_round=None, **kwargs):
        """

        Args:
            X:
            y:
            val_data:
            trial:
            val_round:
            **kwargs:

        Returns:

        """
        self.cat_cols = X.select_dtypes(exclude='number').columns
        X[self.cat_cols] = X[self.cat_cols].astype(float).astype(int)
        if self.model_config.scale == True:
            y = self._scale_y(y, fit=True)
            super().fit(X, y , **kwargs)
        else:
            super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """

        Args:
            X:
            **kwargs:

        Returns:

        """
        X[self.cat_cols] = X[self.cat_cols].astype(float).astype(int)
        pred = super().predict_proba(X, **kwargs)
        pred = pd.DataFrame(pred)
        pred.columns = ['pred0', 'pred1']
        return pred['pred1'].values
    
    def get_params(self, deep=True):
        params = super().get_params(deep)
        cp = copy.copy(self)
        cp.__class__ = RandomForestClassifier
        params.update(cp.__class__.get_params(cp, deep))
        return params