import warnings
warnings.filterwarnings("ignore")

import argparse
import time

from global_utils import file_utils
from data_cleaning import data_clean
from data_preparation import data_prepare
from feature_engineering import feature_module
from training import Model_Trainer
from stacking import Model_Stacker

if __name__ == '__main__':
    print('Let the Pipeline flow!')
    arg_parser = argparse.ArgumentParser(description='Pipeline Orchestration')
    arg_parser.add_argument('--base-path' ,default='../data/' ,help= 'base path at local')
    arg_parser.add_argument('--config',default= '../config/config.yaml' ,help = 'path of config file')
    args = arg_parser.parse_args()

    config = file_utils.load_yaml_config(f'{args.config}')
    
    code_start_time = time.time()
    #Data Cleaning I/O operation
    # data_clean.get_clean_data(args.base_path, config)
    print('Time till Data Cleaning',(time.time() - code_start_time))

    #Data Preparation I/O operation
    # data_prepare.get_prepared_data(args.base_path, config)
    print('Time till Data Preparation',(time.time() - code_start_time))

    #Feature Engineering I/O operation
    # feature_module.create_features(args.base_path, config)
    print('Time till Feature Engineering',(time.time() - code_start_time))

    #Model Training
    # Model_Trainer.train_models(args.base_path, config)
    print('Time till Training',(time.time() - code_start_time))

    #Model Stacking
    Model_Stacker.stack_models(args.base_path, config)
    print('Time till Stacking',(time.time() - code_start_time))