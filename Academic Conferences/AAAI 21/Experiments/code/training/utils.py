import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
import torch
from operator import itemgetter
import torch
import torchcontrib
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA

from global_utils import file_utils
from .Metrics import BCE


def get_usecols(all_feature_groups: str, dtypes: str,model_feature_groups: str, exclude_cols: str, config: str):
    """
        Generates Feature Groups for different models

        Args:
            all_feature_groups: Dictionary of features read from output.json
            dtypes: 
            model_feature_groups: Model specific feature groups
            excludecols: columns to be excluded from the model feature groups
            config: 
        
        Returns:
            Feature columns to be used for that specific model

    """

    usecols = [] 
    usecols_dtypes = {}
    for key in all_feature_groups.keys():
        if key in model_feature_groups:
            usecols_iter = list(set(all_feature_groups[key]) - set(exclude_cols))
            usecols += usecols_iter 
    usecols += [config.data.target_column]
    for key in dtypes.keys():
        if key in usecols:
            usecols_dtypes[key] = dtypes[key]
    return usecols, usecols_dtypes

def get_loader_data(feature_groups: str, feature_group_type: str, model_config, X, val_data = None, fit = None):
    """
        Generates Feature Groups for different models

        Args:
            feature_groups: Dictionary of features read from output.json
            feature_group_type: 
            model_config:
        
        Returns:
            Feature columns to be used for that specific model

    """

    usecols = []
    if fit:
        feature_group_type = model_config.feature_group_type.get(feature_group_type)
        for key in feature_groups.keys():
            if key in feature_group_type:
                usecols_iter = list(set(feature_groups[key]) - set(model_config.exclude_cols))
                usecols += usecols_iter
        return X[usecols], (val_data[0][usecols], val_data[1])
    else:
        feature_group_type = model_config.feature_group_type.get(feature_group_type)
        for key in feature_groups.keys():
            if key in feature_group_type:
                usecols_iter = list(set(feature_groups[key]) - set(model_config.exclude_cols))
                usecols += usecols_iter
        return X[usecols]

def extract_usecols_dtypes(data_path):
    """
    Extract the features and data types of the respective features
    Args:
        data_path:
    """

    usecols_path = f'{data_path}' + '/output.json'
    with open(usecols_path) as f:
        feature_groups = json.load(f)['feature_groups']
    with open(usecols_path) as f:
        dtypes = json.load(f)['dtypes']
    with open(usecols_path) as f:
        cols = json.load(f)['cols']
    return feature_groups, dtypes, cols

def get_model_data(data_path, config, model_config, rounds, usecols=None, dtype=None, model= None, future_type = None, group_level = None, cols = None):
    """
    Get the model data appended in a list
    Args:
        data_path:
        config:
        val_rounds:
        usecols:
        dtype:
        future_type:
        group_level:
    
    """

    cv_data = []
    for i in range(rounds):
        cv_data.append(get_data(data_path, config, model_config, usecols, dtype, model, data_round = f'round{i}', future_type = future_type, group_level =  group_level, cols = cols))
    return dict(cv_data =  cv_data)

def get_data(data_path, config, model_config, usecols=None, dtype=None, model= None, data_round = None, future_type = None, group_level = None, cols = None):
    """
    Read data from the FE path and create the splits
    Args:
        data_path:
        config:
        usecols:
        dtype:
        data_round:
    
    """
    data_list = []
    data_path = f'{data_path}/{group_level}/{future_type}/{data_round}'
    usecols = [*usecols, *config.data.ts_id_columns, config.data.timestamp_column]
    index_col = [*config.data.ts_id_columns , config.data.timestamp_column]
    for split in ['train' , 'val' , 'test', 'future']:
        data = pd.read_csv(file_utils.get_file_object(f'{data_path}/{split}.csv'), names= cols, usecols=usecols, 
                            dtype=dtype , index_col= index_col)
        X = data.drop(columns=config.data.target_column)
        y = data[config.data.target_column].clip(0)
        y = y.astype(float).astype(int)
        data_list.append((X,y))

    return data_list


def create_mpargs(model_data, config, group_level):
    """
    Create multiprocessing args
    Args:
        model_data:
        config:

    """

    group_args = []
    _data = model_data['cv_data'][0][-1][-1]
    group_id_columns =  config.training.groups.get(group_level, {}).group_id_columns
    grouped_data = _data.groupby(group_id_columns)
    group_args.extend((group_id_columns, group_id, data_) for group_id, data_ in grouped_data)
    mp_args = []
    for group_id_columns, group_id, data_ in group_args:
        model_data_dict = {}
        for data_list_key in model_data.keys():
            model_data_list = []
            for data_round in model_data[data_list_key]:
                data_list = []
                for split in data_round:
                    split_X_y = []
                    for data in split:
                        if len(group_id_columns) > 1:
                            for cols, value in zip(group_id_columns, group_id):
                                data = data.loc[(data.index.get_level_values(cols) == value)]
                                if not len(data):
                                    inner_loop_break = True
                                    break
                                else:
                                    inner_loop_break = False
                            if inner_loop_break: break
                        else:
                            for cols, value in zip(group_id_columns, (group_id,)):
                                data = data.loc[(data.index.get_level_values(cols) == value)]
                                if not len(data):
                                    inner_loop_break = True
                                    break
                                else:
                                    inner_loop_break = False
                            if inner_loop_break: break
                        split_X_y.append(data)
                    if inner_loop_break: break
                    data_list.append(split_X_y)
                if inner_loop_break: break
                model_data_list.append(data_list)
            if inner_loop_break: break
            model_data_dict[data_list_key] = model_data_list
        if inner_loop_break: continue
        mp_args.append(model_data_dict)
    return mp_args

def results_to_df(trial_results, model_name, group_level, model_config):
    """
    Convert the trials into dataframe for model shortlisitng
    Args:
        trial_results:
        model_name:

    """

    trial_results = trial_results
    
    ts_metrics = []
    future_pred = pd.DataFrame()

    for i, result in enumerate(trial_results):
        result['trial'] = i + 1
        for data_round in ['cv_data']:
            for _dict in result[data_round]['ts_metrics']:
                ts_metrics.append((_dict.get('ts_id'), result.get('trial'), data_round, _dict.get('_round'), _dict.get('data_split'), _dict.get('metric_value')))
                if _dict['pred']:
                    for pred in _dict['pred']:
                        pred['split'] = _dict.get('data_split')
                        pred['trial'] = result.get('trial')
                        future_pred = pd.concat([future_pred, pred], axis = 0)
    
    cols = ['ts_id', 'trial', 'data_round', '_round', 'data_split', 'metric_value']
    ts_metrics = pd.DataFrame(ts_metrics, columns = cols)
    return ts_metrics, future_pred

def results_empty_df():
    """
    Create Empty Data frames for non valid training datasets
        Args:
            None

    """

    ts_metrics = pd.DataFrame() 
    future_pred = pd.DataFrame()
    ts_metrics_cols = ['ts_id', 'trial', 'data_round', '_round', 'data_split', 'metric_value']
    ts_metrics = pd.DataFrame(ts_metrics, columns = ts_metrics_cols)
    future_pred_cols = ['DC',	'SKU_ID',	'Date',	'actual',	'pred',	'split',	'trial',	'future_type',	'error']
    future_pred = pd.DataFrame(future_pred, columns = future_pred_cols)
    return ts_metrics, future_pred

def seq_len_tail(X, y, config, model_config):
    """
    Subset the data for forecast horizon across splits
    Args:
        X:
        y:
        config:
        model_config:

    """

    X = X.groupby(config.data.ts_id_columns).tail(config.data.forecast_horizon)
    y = y.groupby(config.data.ts_id_columns).tail(config.data.forecast_horizon)
    if model_config.min_obj_size:
        X['object_size'] = X.groupby(config.data.ts_id_columns).size()
        X = X.loc[(X['object_size'] >= config.data.forecast_horizon)].drop(columns = ['object_size'])
        y = pd.DataFrame(y)
        y['object_size'] = y.groupby(config.data.ts_id_columns).size()
        y = y.loc[(y['object_size'] >= config.data.forecast_horizon)].drop(columns = ['object_size'])
        y = y[config.data.target_column].clip(0)
    return X, y



def get_params(self, deep=False): 
    """
    Get parameters
    
    Args:
        
    """
    params = super(xgb.XGBModel, self).get_params(deep=deep)
    if isinstance(self.kwargs, dict):
        params.update(self.kwargs)
    if not params.get('eval_metric', True):
        del params['eval_metric']
    return params
xgb.XGBModel.get_params = get_params

class EarlyStopping:
    """
    
        Early stops the training if validation loss doesn't improve after a given patience
        
        Args:
            patience (int): How long to wait after last time validation loss improved
            verbose (bool): If True, gives a message for each validation loss improvement
            delta (float): Minimum change in the monitored quantity to qualify as an improvement

    """

    def __init__(self, patience, verbose=False, delta=0.00001):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """

            Saves model when validation loss decrease
            Args:
                val_loss:
                model:

        """

        path_to_write = '../data/training/save_checkpoint'
        file_utils.makedirs(path_to_write, exist_ok = True)
        torch.save(model.state_dict(), path_to_write + '/checkpoint.pt')
        self.val_loss_min = val_loss

def parameter_best(model, loss, cur_best):
    """ 
        
    Save state in filename. Also save in best_filename if is_best
    Args:
        model:
        loss:
        cur_best:

    """

    is_best = not cur_best or loss < cur_best
    if is_best:
        cur_best = loss
    return model.state_dict(), cur_best

def parameter_averaging(model, loss, param_config, best_losses, best_states, current_elements_count):
    """

    Save average parameter
    Args:
        model:
        loss:
        param_config:
        best_losses: 
        best_states: 
        current_elements_count:
    """

    window = param_config.parameter_averaging.window
    if current_elements_count == 0:
        best_losses[0] = loss
        best_states[0] = model.state_dict()

    else:
        pos = -1
        for i in range(0, current_elements_count):
            if loss >= best_losses[i]:
                continue
            else:
                pos = i
                break
        if (pos > -1):
            temp_loss = best_losses[pos]
            best_losses[pos] = loss
            temp_state = best_states[pos]
            best_states[pos] = model.state_dict()
            for j in range(window - 1, pos, -1):
                if j == pos + 1:
                    best_losses[j] = temp_loss
                    best_states[j] = temp_state
                else:
                    best_losses[j] = best_losses[j-1]
                    best_states[j] = best_states[j-1]
        elif current_elements_count < window:
            best_losses[current_elements_count] = loss
            best_states[current_elements_count] = model.state_dict()
    
    

    if current_elements_count == window:
        averaged_state = best_states[window - 1]
        for key in averaged_state:
            for i in range(window-1, -1, -1):
                averaged_state[key] = (averaged_state[key] + best_states[i][key])/2
    else:
        averaged_state =  best_states[0]
    
    if current_elements_count < window:
        current_elements_count+=1

    return averaged_state, current_elements_count
    
def save_best_checkpoint(averaged_state):
    """

        Saves model when validation loss decrease
        Args:
            averaged_state:

    """

    path_to_write = '../data/training/save_checkpoint'
    file_utils.makedirs(path_to_write, exist_ok = True)
    torch.save(averaged_state, path_to_write + '/best.pt')

def compression_multiplier(cont_in_features):
    """
    Multiplier of the compression factor for Deep Neural Net
    Args:
        cont_in_features:

    """

    i = 1
    if cont_in_features <= 64:
        i = 1
    elif cont_in_features <= 256:
        i = 2
    else:
        i = 4
    return i

def DL_model_params(parameters, param_config, optimizer, scheduler, SWA, train_loader):
    """
    Updates the DL model parameters including optimizer, scheduler and earlystopper
    Args:
        parameters:
        param_config:
        optimizer:
        scheduler:
        SWA:

    """

    scheduler_name = scheduler
    if optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(parameters, **param_config.optimizer_params.RMSprop)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, **param_config.optimizer_params.SGD)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, **param_config.optimizer_params.Adam)
    else:
        optimizer = torch.optim.RMSprop(parameters, **param_config.optimizer_params.RMSprop)
    
    if SWA:
        param_config.scheduler_params.CyclicLR.cycle_momentum = False
        optimizer = torchcontrib.optim.SWA(optimizer, **param_config.SWA_params)
    
    if scheduler == 'CyclicLR':
        param_config.scheduler_params.CyclicLR.step_size_up = len(train_loader) * param_config.scheduler_params.CyclicLR.step_size_up 
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **param_config.scheduler_params.CyclicLR)
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **param_config.scheduler_params.ReduceLROnPlateau)
    else:
        param_config.scheduler_params.CyclicLR.step_size_up = len(train_loader) * param_config.scheduler_params.CyclicLR.step_size_up 
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **param_config.scheduler_params.CyclicLR)

    early_stopper = EarlyStopping(patience= param_config.callback_params.earlystopping.patience)

    return optimizer, scheduler, scheduler_name, early_stopper

def regularizer(param_config, parameters):
    """
    Regularize the parameters
    Args:
        param_config:
        parameters:

    """

    if param_config.regularization.l2_norm:
        reg_loss = param_config.regularization.reg_param.reg_loss
        for param in parameters:
            if reg_loss:
                reg_loss = reg_loss + param.norm(2)**2
            else:
                reg_loss += torch.sum(param**2)
        return param_config.regularization.reg_param.lmbda * reg_loss
    else:
        return 0

def loss_func(config):
    """
    loss function based on config
    Args:
        config:

    """

    if config.training.Metric == 'BCELoss':
        return nn.BCEWithLogitsLoss()