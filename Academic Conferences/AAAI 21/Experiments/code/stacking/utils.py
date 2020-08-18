import os
import pandas as pd
import numpy as np
import copy
from copy import deepcopy

from global_utils import file_utils 
from training.Metrics import BCE

def top_candidate_metrics(data_path, config, future_type):
    """
    Input all the model metrics
    Args:
        data_path:
        config:
        future_type:
    
    """

    ts_candidates = pd.DataFrame()
    for subdir, dirs, files in os.walk(data_path):
        for _file in files:
            if _file == 'ts_metrics.csv':
                ts_metrics_path = os.path.join(subdir, _file)
                if ts_metrics_path.rsplit('/',3)[-3] == future_type:
                    ts_metrics = pd.read_csv(file_utils.get_file_object(f'{ts_metrics_path}'))
                    if config.stacking.round_penalty:
                        ts_metrics['metric_value'] = ts_metrics['metric_value']/(ts_metrics['_round'] + 1)
                    ts_metrics =  pd.DataFrame(ts_metrics.loc[(ts_metrics['data_round'] == 'cv_data') & (ts_metrics['data_split'] == 'test')]
                                    .groupby(['ts_id', 'trial'])['metric_value']
                                    .agg(np.mean)
                                    .reset_index()
                                    .assign(model_name = ts_metrics_path.rsplit('/',2)[-2], level = ts_metrics_path.rsplit('/',4)[-4] ))
                    ts_metrics['model_key'] = ts_metrics[['level','model_name','trial']].astype(str).apply('_'.join,1)
                    ts_metrics['model_level'] = ts_metrics[['level','model_name']].astype(str).apply('_'.join,1)
                    if config.stacking.model_exclude:
                        ts_metrics = ts_metrics[~ts_metrics.model_level.isin(config.stacking.model_exclude)]
                    if config.stacking.model_include:
                        ts_metrics = ts_metrics[ts_metrics.model_level.isin(config.stacking.model_include)]
                    ts_metrics   = ts_metrics.drop(['level', 'model_name', 'trial', 'model_level' ], axis=1)
                    ts_candidates = pd.concat([ts_candidates, ts_metrics], axis = 0)
    
    ts_candidates['metric_value'] = ts_candidates['metric_value'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ts_candidates['candidate_rank'] = ts_candidates.groupby('ts_id')['metric_value'].rank(method = 'first', ascending=True).astype(int)
    ts_candidates = ts_candidates[(ts_candidates['candidate_rank'] <= config.stacking.n_candidates)]
    ts_candidates['candidate_weights'] = ts_candidates.apply(lambda row: weight_alloc(row['metric_value'], row['candidate_rank']), axis=1)
    return ts_candidates

def weight_alloc(value, rank):					
        if (value >= 1 ) & (rank == 1):
            weight = 1.0
        elif (value >= 1) & (rank > 1) :
            weight = 0.0
        else:
            weight = 1.0 - value
        return weight

def top_candidate_data(top_candidates, data_path, config, future_type):
    """
    Input top candidate datasets for the candidate models
    Args:
        top_candidates:
        data_path:
        config:
        future_type:

    """
    
    future_candidates_df = pd.DataFrame()

    for subdir, dirs, files in os.walk(data_path):
        for _file in files:
            if _file in ['future.csv']:
                _path = os.path.join(subdir, _file)
                if _path.rsplit('/',3)[-3] == future_type:
                    candidate_data = pd.read_csv(file_utils.get_file_object(f'{_path}'))
                    candidate_data['ts_id'] = candidate_data[config.data.ts_id_columns].astype(str).apply('_'.join,1)
                    candidate_data = candidate_data.assign(model_name = _path.rsplit('/',2)[-2], level = _path.rsplit('/',4)[-4] )
                    candidate_data['model_key'] = candidate_data[['level','model_name','trial']].astype(str).apply('_'.join,1)
                    candidate_data   = candidate_data.drop(['level', 'model_name', 'trial' ], axis=1)
                    selected_candidates = pd.merge(candidate_data, top_candidates, on = ['ts_id', 'model_key'], how = 'inner')
                    if config.stacking.NewProduct:
                        leftover_candidates = candidate_data[~candidate_data['ts_id'].isin(top_candidates['ts_id'])]
                        leftover_candidates['metric_value'] = 1
                        leftover_candidates['candidate_rank'] = 1
                        leftover_candidates['candidate_weights'] = 1
                        selected_candidates = pd.concat([selected_candidates, leftover_candidates])
                    cols =  ['ts_id', 'actual', 'pred', 'split', 'candidate_rank', 'candidate_weights'] + [config.data.timestamp_column]
                    selected_candidates = selected_candidates[cols]
                    future_candidates_df = pd.concat([future_candidates_df, selected_candidates], axis = 0)

    future_candidate_list = []
    for split in ['train', 'val', 'test', 'future']:
        data_split = future_candidates_df[future_candidates_df['split'] == split]
        future_candidate_list.append(data_split)
    
    return dict(future_data = future_candidate_list)


def create_mpargs(model_data, config):
    """
    Create multiprocessing args
    Args:
        model_data:
        config:

    """

    group_args = []
    _data = model_data['future_data'][-1]
    group_id_columns = 'ts_id'
    grouped_data = _data.groupby(group_id_columns)
    group_args.extend((group_id_columns, group_id, data_) for group_id, data_ in grouped_data)
    
    mp_args = []
    for group_id_columns, group_id, data_ in group_args:
        model_data_dict = {}
        for data_list_key in model_data.keys():
            data_list = []
            for split in model_data[data_list_key]:
                index_cols = ['ts_id'] + [config.data.timestamp_column]
                split = split.loc[(split[group_id_columns] == group_id)]
                X = (split.drop(columns= ['actual'])
                        .pivot(index= config.data.timestamp_column, columns='candidate_rank', values='pred')
                        .reset_index()
                        .assign(ts_id = group_id)
                        .set_index(index_cols)
                        .fillna(0))
                y = (split.drop(columns= ['pred'])
                        .groupby(index_cols)
                        .agg({'actual': 'mean'}))
                data_list.append([X,y])
            model_data_dict[data_list_key] = data_list
        mp_args.append(model_data_dict)
    return mp_args

def generate_predictions(split, data, config):
    """
    Predictions generator for K_best
    Args:
        split:
        data:
        config:

    """

    data = data.loc[(data['candidate_rank'] <= config.stacking.k_best_candidates)]
    cols =  ['ts_id'] + [config.data.timestamp_column]
    if config.stacking.candidate_weight:
        data['pred'] = data['pred'] * data['candidate_weights']
        data = (data.groupby(cols)
                .agg({'actual': 'mean', 'pred' : 'sum', 'candidate_weights': 'sum'}))
        data['pred'] = data['pred']/data['candidate_weights']
        data = data[['actual', 'pred']]
    else:
        data = (data.groupby(cols)
                .agg({'actual': 'mean', 'pred' : 'mean'}))
    return data

def metric_cal(split, predictions_data, future_type):
    """
    Calculate K_best metric for all splits
    Args:
        predictions_data:
        future_type:

    """

    future_predictions = copy.deepcopy(predictions_data)
    print('future_predictions', future_predictions)
    metric = []
    if (future_type =='future') & (split == 'future'):
        metric.append((split, 0))
    else:
        metric.append((split, BCE(future_predictions['actual'], future_predictions['pred'])))
    metric = pd.DataFrame(metric, columns = ['data_split', 'metric_value'])
    return metric