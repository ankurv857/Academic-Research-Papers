import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any
import json
from os import walk
import multiprocessing as mp
from functools import partial
import random

from global_utils import file_utils
from training.Base_Models import BaseModel
from training.Metrics import wmape
from training.hyperparameter import HyperOpt
from . import utils


def train_models(base_path: str, config: str, **kwargs):
    """
    Main entry point to train the models

    Args:

        base_path: path of the input data 
        config: global config
        **kwargs:Keyword Arguments

    Returns:

    """
    

    data_path = f"{base_path}feature_engg"
    val_rounds = config.training.val_rounds

    trainer = Trainer(data_path, val_rounds, config)
    trainer.train()


@dataclass
class Trainer:
    data_path: str
    val_rounds: int
    config: Any

    def train(self,  **training_params):
        self.feature_groups, self.dtypes, self.cols = utils.extract_usecols_dtypes(self.data_path)
        for group_level in self.config.training.groups:
            for future_type in self.config.training.future_types:
                self.group_level = group_level
                self.future_type = future_type
                self.train_models()

    def train_models(self):
        """
        Run the models
        Args:
            None:

        """

        self.group_config = self.config.training.groups.get(self.group_level, {})
        if self.group_config.model_names:
            for model_name in self.group_config.model_names:
                self.model_name = model_name
                self.model = BaseModel.from_name(model_name)(self.config, self.feature_groups, self.dtypes, model_name)
                self.model_config = self.config.training.models.get(model_name, {})
                model_data = utils.get_model_data(self.data_path, self.config, self.model_config, self.val_rounds, self.model.usecols, self.model.dtypes, 
                                                self.model, future_type = self.future_type, group_level = self.group_level, cols = self.cols)
                if self.group_config.group_id_columns:
                    mp_args = utils.create_mpargs(model_data, self.config, self.group_level)
                    if self.config.training.n_groups:
                        mp_args = mp_args[:self.config.training.n_groups]
                    if self.model_config.multiprocessing == True:
                        ts_metrics_df, future_pred_df = self.multiprocess_args(mp_args)
                    else:
                        ts_metrics_df = pd.DataFrame()
                        future_pred_df = pd.DataFrame()
                        for args in mp_args:
                            ts_metrics, future_pred =  self.cv_models(args)
                            ts_metrics_df = pd.concat([ts_metrics_df, ts_metrics], axis = 0).reset_index(drop = True)
                            future_pred_df = pd.concat([future_pred_df, future_pred], axis = 0)
                else:
                    ts_metrics_df, future_pred_df = self.cv_models(model_data)

                path_to_write = f'{self.data_path.rsplit("/",1)[0]}/training/{self.group_level}/{self.future_type}/{self.model_name}'
                file_utils.makedirs(path_to_write, exist_ok = True)
                for df_name, df in zip([ts_metrics_df, future_pred_df], ['ts_metrics', 'future']):
                    df_name.to_csv(f'{path_to_write}/{df}.csv', index = True)


    def cv_models(self, model_data):
        """
        Cross Validation models
        Args:
            model_data:
        
        """
        if self.model.model_criteria(model_data, self.model_config):
            if self.model_config.get('hyperparams'):
                cv_model = HyperOpt(self.model, 'hyperopt', self.model_config.hyperparams, self.config, self.model_config)
                trial_params = cv_model.optimize(model_data, max_evals= self.model_config.max_evals)
                trial_results = []
                for _params in trial_params:
                    self.model.set_params(**_params['params'])
                    metrics_dict = self.model.cross_validate(model_data, self.config, self.model_config)
                    metrics_dict['trial'] = _params['trial']
                    trial_results.append(metrics_dict)
            elif self.model_config.get('params'):
                trial_results = []
                if self.model_config.hyperparam_trials:
                    if self.model_config.random_sample:
                        self.model_config.trial_params = random.sample(self.model_config.trial_params, len(self.model_config.trial_params))
                    for trial, _params in enumerate(self.model_config.trial_params):
                        if trial + 1  <= self.model_config.trials:
                            metrics_dict = self.model.cross_validate(model_data, self.config, self.model_config, **_params)
                            metrics_dict['trial'] = trial + 1
                            trial_results.append(metrics_dict)
                else:
                    for trial in range(self.model_config.trials):
                        _params = self.model_config.default_params
                        metrics_dict = self.model.cross_validate(model_data, self.config, self.model_config, **_params)
                        metrics_dict['trial'] = trial + 1
                        trial_results.append(metrics_dict)
            return utils.results_to_df(trial_results, self.model_name, self.group_level, self.model_config)
        else:
            return utils.results_empty_df()
    
    def multiprocess_args(self, mp_args):
        """
        Multiprocessing function
        Args:
            mp_args:

        """

        if len(mp_args) >= 1:
            n_cpus = self.config.get('n_cpus')
            if not n_cpus:
                n_cpus = .8
            if isinstance(n_cpus, float):
                n_cpus = int(mp.cpu_count() * n_cpus)
            pool = mp.Pool(n_cpus)
            outputs = pool.map(partial(self.cv_models), mp_args)
            pool.close()
            pool.join()

            ts_metrics_df = pd.DataFrame()
            future_pred_df = pd.DataFrame()
            for output in outputs:
                ts_metrics, future_pred = output
                ts_metrics_df = pd.concat([ts_metrics_df, ts_metrics], axis = 0).reset_index(drop = True)
                future_pred_df = pd.concat([future_pred_df, future_pred], axis = 0)
        return ts_metrics_df, future_pred_df