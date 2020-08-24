import pandas as pd
import sys
import os

from global_utils import file_utils

def get_prepared_data(base_path: str, config: str, **kwargs):
    """

        I/O operation for Data Prep module
        
        Args:

            base_path: path of the input data 
            config: YAML config file from config location
            **kwargs: Keyword Arguments

        Returns:
            Prepared Data at Prepared data location

    """

    file_path = base_path + config.data_cleaning.cleaned_data_path
    input_files = file_utils.walk(file_path)
    data_dict = dict()

    function_dict = {
        'aggregate_data' : aggregate_data,
        'join_data': join_data,
        'append_data': append_data,
        'column_rename': column_rename,
        'exclude_columns': exclude_columns,
        'concat_columns': concat_columns,
        'split_column': split_column,
        'melt_data': melt_data,
        'new_columns': new_columns}

    serial_ops = config.data_preparation.serial_ops
    if serial_ops:
        for serial_op in serial_ops:
            data = get_dataframe(data_dict, input_files, serial_op.input.name, serial_op.input.file_read_kwargs)
            ops = serial_op.operations

            if ops:
                for op in ops:
                    if 'kwargs' not in op.keys() and 'args' not in op.keys():
                        data = function_dict[op.operation](data_dict, input_files, data)
                    elif 'kwargs' not in op.keys():
                        data = function_dict[op.operation](data_dict, input_files, data, *op.args)
                    elif 'args' not in op.keys():
                        data = function_dict[op.operation](data_dict, input_files, data, **op.kwargs)
                    else:
                        data = function_dict[op.operation](data_dict, input_files, data, *op.args, **op.kwargs)
            
            path_to_write = f'{base_path}{config.data_preparation.data_prep_path}'
            file_utils.makedirs(path_to_write, exist_ok = True)
            _filename = path_to_write + '/' + f'{serial_op.output}.csv'
            data.to_csv(_filename, index=False)


def get_dataframe(data_dict , input_files, obj , kwargs = None):
    """

        Returns dataframe object given input.
        Args:
            data_dict: Dictionary of DataFrames -> Dict
            input_files: list of available data source files -> List
            obj: The object to be retrieved as DataFrame -> pd.DataFrame
            kwargs: Any kwargs to be applied while reading data to pandas object

        Returns:
            Object as DataFrame

    """

    if type(obj) == str:
        if obj in data_dict.keys():
            return data_dict[obj]
        else:
            kwargs = kwargs or {}
            return pd.read_csv(file_utils.get_file_object(extract_file_on_pattern(obj, input_files)),**kwargs)
    elif type(obj) == pd.DataFrame:
        return obj


def extract_file_on_pattern(pattern: str, input_files: str):
    """

    Returns the name of the file with given pattern from list of input files.
    There should be single file with given pattern
    Args:
        pattern: Pattern of the input file
        Input Files: List of input files

    Returns:
        Returns the name of the file with given pattern from list of input files.

    """

    files_with_pattern = [file for file in input_files if pattern in file.rsplit('/')[-1]]
    return files_with_pattern[0]


def aggregate_data(data_dict, input_files: str , dataset: str , grouping_columns: str , aggregations: str):
    """

        Aggregates data by given column and gets the required metrics
        Args:

            data_dict: Data dictionary
            input_files: List of input files -> List[str]
            dataset: input dataset ->  Union[str, pd.DataFrame]
            grouping_columns: List of group by column names -> List[str]
            aggregations: Dictionary of operations to be performed. Key being column name and value being the name of the operation -> Dict[str, str]
            ex:
                aggregations:
                    Age: min
                    Weight: max
        Returns: Aggregated dataframe

    """

    df = get_dataframe(data_dict, input_files, dataset)
    grouped_df = df.groupby(grouping_columns, as_index=False).agg(aggregations)
    return grouped_df


def join_data(data_dict, input_files: str, file1: str, file2: str, file1_col: str, file2_col: str, join_type: str, drop_col: str):
    """
        Performs join operation on given 2 datasets
        Args:

            data_dict: Data dictionary
            input_files: List of input files -> List[str]
            file1: file1 input dataset ->  Union[str, pd.DataFrame]
            file2: file1 input dataset ->  Union[str, pd.DataFrame]
            file1_col: column list of file for merge -> List[str]
            file2_col: column list of file for merge -> List[str]
            join_type: type of join -> str = 'inner'
            drop_col: column to be dropped after join -> List[str] = None

        Returns:
            Joined Dataframe -> pd.DataFrame

    """

    df1 = get_dataframe(data_dict, input_files, file1)
    df2 = get_dataframe(data_dict, input_files, file2)
    result_df = df1.merge(df2, left_on=file1_col, right_on=file2_col, how=join_type)
    if drop_col is not None:
        result_df = result_df.drop(columns=drop_col)
    return result_df


def append_data(data_dict, input_files: str, file1: str, datasets) :
    """

        Appends given datasets
        Args:
            data_dict: Dictionary of dataframes -> Dict
            input_files: List of possible source files -> List[str]
            file1: First dataset -> Union[str, pd.DataFrame]
            datasets: List of datasets -> List
        Return:
            Appended DataFrame -> pd.DataFrame

    """
    datasets = [file1, *datasets]
    result_df = pd.concat((get_dataframe(data_dict, input_files, dataset) for dataset in datasets), ignore_index=True)
    return result_df


def column_rename(data_dict, input_files: str, dataset: str, rename_dict):
    """

        Renames the columns and returns the dataframe
        Args:
            data_dict: Dictionary of dataframes -> Dict
            input_files: List of possible source files -> List[str]
            dataset: dataset for the column rename -> Union[str, pd.DataFrame]
            rename_dict: rename column -> *to be ensured
        Return:
            DataFrame with renamed columns -> pd.DataFrame
    
    """

    df = get_dataframe(data_dict, input_files, dataset)
    df = df.rename(columns=rename_dict)
    return df

def exclude_columns(data_dict , input_files:str, dataset: str, columns_list) :
    """

        Returns dataframe after excluding the columns
        Args:
            data_dict: Dictionary of dataframes -> Dict
            input_files: List of possible source files -> List[str]
            dataset: dataset for the column rename -> Union[str, pd.DataFrame]
            columns_list: list of columns -> list
        Return:
            DataFrame with renamed columns -> pd.DataFrame
    
    """

    df = get_dataframe(data_dict, input_files, dataset)
    columns = list(set(df.columns) - set(columns_list))
    return df[columns]

def concat_columns(data_dict , input_files:str, dataset: str, new_column, columns_list) :
    """

        Returns dataframe after excluding the columns
        Args:
            data_dict: Dictionary of dataframes -> Dict
            input_files: List of possible source files -> List[str]
            dataset: dataset for the column rename -> Union[str, pd.DataFrame]
            new_column:
            columns_list: list of columns -> list
        Return:
            DataFrame with renamed columns -> pd.DataFrame
    
    """

    df = get_dataframe(data_dict, input_files, dataset)
    for col in columns_list[1:]:
        df[new_column] = df[columns_list[0]] + '_' + df[col]
    return df


def split_column(data_dict, input_files: str, datasource: str, source_column_name: str, new_columns: str, split_pattern, n_splits=-1, drop_source_column=True):
    """

    Splits given column into specified number of columns based on split pattern.
    Args:
        data_dict: Dictionary of dataframes
        input_files: List of possible source files -> List[str]
        datasource: Datasource name -> Union[str, pd.DataFrame]
        source_column_name: column name of the datasource which has to be split
        new_columns: list of new column names -> List[str]
        split_pattern: split pattern/character
        n_splits: Number of splits to be created
        drop_source_column: If source column should be dropped or not

    Result:
        DataFrame with splitted column -> pd.DataFrame:

    """

    assert n_splits == len(new_columns), "Number of new column names should be equal to number of splits"
    df = get_dataframe(data_dict, input_files, datasource)
    df[new_columns] = df[source_column_name].str.split(split_pattern, n=n_splits - 1, expand=True)
    if drop_source_column:
        df.drop(columns=[source_column_name], inplace=True)
    return df

def melt_data(data_dict , input_files:str, dataset: str, id_columns, melt_column) :
    """

        Returns dataframe after melting
        Args:
            data_dict: Dictionary of dataframes -> Dict
            input_files: List of possible source files -> List[str]
            dataset: dataset for the column rename -> Union[str, pd.DataFrame]
            id_columns:
            melt_column:
        Return:
            DataFrame with melted columns -> pd.DataFrame
    
    """

    df = get_dataframe(data_dict, input_files, dataset)
    df_melted = pd.melt(df, id_vars= id_columns , var_name=melt_column)
    return df_melted

def new_columns(data_dict , input_files:str, dataset: str, column_name, value) :
    """
        Returns dataframe after adding the column with the given value
        Args:
            data_dict: Dictionary of dataframes -> Dict
            input_files: List of possible source files -> List[str]
            dataset: dataset for the column rename -> Union[str, pd.DataFrame]
            column_name:
            value:
        Return:
            DataFrame with additional columns -> pd.DataFrame
    
    """

    df = get_dataframe(data_dict, input_files, dataset)
    df[column_name] = value
    return df