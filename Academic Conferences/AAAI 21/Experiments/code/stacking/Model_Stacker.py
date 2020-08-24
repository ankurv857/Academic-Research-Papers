import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any
import multiprocessing as mp
from functools import partial

from stacking.Models import BaseModel, K_best
from global_utils import file_utils 
from . import utils

def stack_models(base_path: str, config: str, **kwargs):
    """
    Main entry point to stacking of models

    Args:

        base_path: path of the input data 
        config: global config
        **kwargs:Keyword Arguments

    Returns:

    """
    

    data_path = f"{base_path}training"

    stacker = Stacker(data_path, config)
    stacker.stack()

@dataclass
class Stacker:
    data_path: str
    config: Any

    def stack(self, **stacking_params):
        for future_type in self.config.stacking.future_types:
            self.future_type = future_type
            top_candidates = utils.top_candidate_metrics(self.data_path, self.config, future_type)
            model_data = utils.top_candidate_data(top_candidates, self.data_path, self.config, future_type)
            for model_name in self.config.stacking.model_names:
                if model_name == 'K_best':
                    default_model = K_best(model_data, self.config)
                    future_predictions, metric = default_model.model_compute(future_type)
                else:
                    self.model = BaseModel.from_name(model_name)(self.config)
                    self.model_config = self.config.stacking.models.get(model_name, {})
                    mp_args = utils.create_mpargs(model_data, self.config)
                    if self.config.stacking.n_groups:
                        mp_args = mp_args[:self.config.stacking.n_groups]
                    if self.model_config.multiprocessing == True:
                        future_predictions, metric = self.multiprocess_args(mp_args, future_type)

                print(model_name,metric)
                path_to_write = f'{self.data_path.rsplit("/",1)[0]}/stacking/{future_type}/{model_name}'
                file_utils.makedirs(path_to_write, exist_ok = True)
                for df_name, df in zip([top_candidates, future_predictions, metric], 
                                        ['top_candidates', 'future_predictions', 'metric']):
                    df_name.to_csv(f'{path_to_write}/{df}.csv', index = True)
    
    def _models(self, model_data):
        """
        models of stacker
        Args:
            model_data:
        
        """
        
        if self.model_config.get('params'):
            return self.model.run_stacker(model_data, self.config, self.future_type)
    
    def multiprocess_args(self, mp_args, future_type):
        """
        Multiprocessing function
        Args:
            mp_args:
            future_type:

        """

        if len(mp_args) >= 1:
            n_cpus = self.config.get('n_cpus')
            if not n_cpus:
                n_cpus = .8
            if isinstance(n_cpus, float):
                n_cpus = int(mp.cpu_count() * n_cpus)
            pool = mp.Pool(n_cpus)
            outputs = pool.map(partial(self._models), mp_args)
            pool.close()
            pool.join()

            future_predictions = pd.DataFrame()
            for output in outputs:
                future_predictions = pd.concat([future_predictions, output], axis = 0)
            if future_type == 'virtual_future':
                metric = []
                metric.append(('test',sum(np.abs(future_predictions['actual'] - future_predictions['pred']))/sum(future_predictions['actual'])))
                metric = pd.DataFrame(metric, columns = ['data_split', 'metric_value'])
            else:
                metric = pd.DataFrame()
        return future_predictions, metric