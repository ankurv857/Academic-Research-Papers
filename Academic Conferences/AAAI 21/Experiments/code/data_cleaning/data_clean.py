import pandas as pd
import numpy as np
import os
import csv
import datetime
from global_utils import file_utils

def get_clean_data(base_path: str, config: str, **kwargs):
    """

        I/O operation of Data Clean module
        Args:
            base_path: base path of the data
            config: YAML config file from config location
            **kwargs: Keyword Arguments
        
        Returns:
            CLeaned Data at Data Clean location

    """

    if config.data_cleaning.dataframe_attribues:
        dataframe_attribues = config.data_cleaning.dataframe_attribues
    else:
        dataframe_attribues = []
    raw_path = base_path + config.data_cleaning.raw_data_path
    df_list = load_csv(raw_path)
    
    for df in df_list:
        output_df = df_list[df] 

        if df in dataframe_attribues:
            for df_info in dataframe_attribues[df]:

                type_name = df_info['type'] 

                if type_name == 'remove_sc':
                    column_name = df_info['column_name'] 
                    original_string = df_info['original_string'] 
                    new_string = df_info['new_string'] 
                    output_df = replace_strings(output_df, column_name, original_string, new_string)

                elif type_name == 'coerce_cols':
                    column_name = df_info['column_name']
                    output_df = coerce_cols(output_df, column_name)

                elif type_name == 'concat_cols':
                    column_name = df_info['column_name']
                    output_df = concat_cols(output_df, column_name)

                elif type_name == 'dateformat_correction':
                    date_format = df_info['date_format']
                    output_df = dateformat_correction(output_df, date_format)

                elif type_name == 'month_todate':
                    new_date_column = df_info['new_date_column']
                    month_column = df_info['month_column']
                    output_df = create_date_from_month(output_df, new_date_column, month_column)

                elif type_name == 'month_and_year_todate':
                    new_date_column = df_info['new_date_column']
                    year_col = df_info['year_col']
                    month_col = df_info['month_col']
                    output_df = create_date_from_month_and_year(output_df, new_date_column, year_col, month_col)

                elif type_name == 'yearmonth_todate':
                    new_date_column = df_info['new_date_column']
                    month_col = df_info['month_column']
                    date_format = df_info['date_format']
                    output_df = create_date_from_yearmonth(output_df, new_date_column, month_col, date_format)
                
                elif type_name == 'subset_data':
                    column_name = df_info['column_name']
                    condition = df_info['condition']
                    value = df_info['value']
                    output_df = subset_data(output_df, column_name, condition, value)

        path_to_write = f'{base_path}{config.data_cleaning.cleaned_data_path}'
        file_utils.makedirs(path_to_write, exist_ok = True)
        _filename = path_to_write + '/' + df
        output_df.to_csv(_filename, index=False)


def update_header_csv(file_path: str, header_file_path: str):
    """
       
        Updates Header of the csv.
        Args:
            file_path: csv file name along with location
            header_file_path: header file name along with location

        Returns:
            Pandas Dataframe with updated header

    """

    with open(header_file_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    df = pd.read_csv(file_path, names = header, header=1)
    return df

def load_csv(file_path: str):
    """
       
        Load all csv files from actual path into a dictionary with file name as key.
        Args:
            file_path: file path for a project in raw data folder
        
        Returns:
            Dictionary with filename as key and pandas dataframe of that csv as value

    """

    df_list = {}
    #Stores all file paths in a file list
    files = file_utils.walk(file_path)
    for file_name in files:
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(file_name))
            file_name_csv = os.path.basename(file_name)
            df_list[file_name_csv] = df
    return df_list

def replace_na_with_null(df):
    """

        Replaces NA with Null for all the columns in the DataFrame
        Args:
            df: Pandas DataFrame

        Returns:
            DataFrame with all Nan values replaced by Null

    """

    df = df.replace(np.nan, '', regex = True)
    return df


def replace_strings(df, column_name: str, original_string: str, new_string: str):
    """

        Replaces the special characters with none
        Args:
            df: Pandas DataFrame
            column_name: column name of csv with special chars
            original_string: string which needs to be replaced
            new_string: to be replaced to

        Return:
            DataFrame with replaced special chars

    """

    df[column_name] = df[column_name].str.replace(original_string,new_string)
    return df

def dateformat_correction(df, date_format):
    """

        Correct the dateformat of the imported files
        Args:
            df: Pandas DataFrame
            date_format:

        Returns:
            DataFrame with standard date format yyyy-mm-dd

    """

    df_raw = df
    for col in df.columns:
        if df[col].dtype =='object':
            try:
                df[col] = pd.to_datetime(df[col], format = date_format)
            except ValueError:
                pass
    return df_raw


def create_date_from_month(df, new_date_column: str, month_col: str):
    """

        Correct the dateformat of the imported files
        Args:
            df: Pandas DataFrame
            new_date_column: 
            month_col: 

        Returns:
            DataFrame with standard date format yyyy-mm-dd

    """

    df[new_date_column] = df[month_col].apply(lambda x: f"{x}01")
    return df


def create_date_from_month_and_year(df, new_date_column: str, year_col: str, month_col: str):
    """

        Correct the dateformat of the imported files
        Args:
            df: Pandas DataFrame
            new_date_column: 
            year_col: 
            month_col: 

        Returns:
            DataFrame with standard date format yyyy-mm-dd

    """

    df[new_date_column] = df[year_col].map(str) + df[month_col].map("{:02}".format) + "01"
    return df

def create_date_from_yearmonth(df, new_date_column: str, date_col: str, date_format: str):
    """

        Correct the dateformat of the imported files
        Args:
            df: Pandas DataFrame
            new_date_column: 
            date_col: 
            date_format: 

        Returns:
            DataFrame with standard date format yyyy-mm-dd

    """

    df[new_date_column] = df[date_col].astype(str).str[:4].astype(str) + '-' + df[date_col].astype(str).str[4:].astype(str) + '-' + '01' 
    df[new_date_column] = pd.to_datetime(df[new_date_column] , format = date_format)
    return df

def coerce_cols(df, column_name):
    """

        Coerces the column
        Args:
            df: Pandas DataFrame
            column_name: 

        Returns:
            Coerces the column

    """

    df[column_name] = pd.to_numeric(df[column_name], errors = 'coerce').astype(float)
    return df

def concat_cols(df, column_name):
    """

        Coerces the column
        Args:
            df: Pandas DataFrame
            column_name: 

        Returns:
            Coerces the column

    """

    df[column_name] = df[column_name].astype(str) + '_'
    return df

def subset_data(df, column_name, condition, value):
    """

        Subset the data
        Args:
            df: Pandas DataFrame
            column_name: 
            condition:
            value:

        Returns:
            Data subseted

    """

    if condition == '<':
        df = df[(df[column_name] < value)]
    elif condition == '>':
        df = df[(df[column_name] > value)]
    return df