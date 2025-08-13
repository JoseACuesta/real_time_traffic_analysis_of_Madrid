import polars as pl

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import os
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def split_train_validation_and_test_data(
    df: pl.DataFrame, train_validation_data_path: Path, test_data_path: Path
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series]:
    """
    Splits the input DataFrame into train/validation and test datasets based on the 'year' column,
    saves them as Parquet files if they do not already exist, and returns the relevant datasets.
    If the Parquet files specified by `train_validation_data_path` and `test_data_path` do not exist,
    the function splits the data such that all rows with 'year' == 2024 are used as test data,
    and the rest as train/validation data. The splits are saved to the provided file paths.
    If the files exist, they are loaded directly.
    :param df: The input Polars DataFrame containing the data to be split.
    :type df: pl.DataFrame
    :param train_validation_data_path: The file path where the train/validation data Parquet file is or will be stored.
    :type train_validation_data_path: Path
    :param test_data_path: The file path where the test data Parquet file is or will be stored.
    :type test_data_path: Path
    :returns: 
        - test_data (pl.DataFrame): The test dataset (rows where 'year' == 2024).
        - train_validation_data (pl.DataFrame): The train/validation dataset (rows where 'year' != 2024).
        - X_test (pl.DataFrame): The test features (test_data without the 'carga' column).
        - y_test (pl.Series): The test target variable (the 'carga' column from test_data).
    :rtype: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series]
    """
    
    if not os.path.exists(train_validation_data_path) or not os.path.exists(test_data_path):
        test_data = df.filter(pl.col('year') == 2024)
        train_validation_data = df.filter(pl.col('year') != 2024)

        test_data.write_parquet(file=test_data_path)
        train_validation_data.write_parquet(file=train_validation_data_path)
    
    train_validation_data = pl.read_parquet(source=train_validation_data_path)
    test_data = pl.read_parquet(source=test_data_path)

    X_test = test_data.drop('carga')
    y_test = test_data['carga']
    
    return test_data, train_validation_data, X_test, y_test

def split_train_and_validation_data(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """
    Splits the input DataFrame into train and validation datasets based on the 'year' column.
    :param df: The input Polars DataFrame containing the data to be split.
    :type df: pl.DataFrame
    :returns: 
        - X_train (pl.DataFrame): The train features (rows where 'year' != 2023).
        - X_val (pl.DataFrame): The validation features (rows where 'year' == 2023).
        - y_train (pl.Series): The train target variable.
        - y_val (pl.Series): The validation target variable.
    :rtype: tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]
    """
    
    is_2023 = df['year'] == 2023

    X = df.drop('carga')
    y = df['carga']

    X_train = X.filter(~is_2023)
    X_val = X.filter(is_2023)

    y_train = y.filter(~is_2023)
    y_val = y.filter(is_2023)

    return X_train, X_val, y_train, y_val

def normalize_and_scale_train_and_val_data(X_train: pl.DataFrame, X_val: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Normalizes and scales the training and validation datasets by applying one-hot encoding to categorical columns
    and standard scaling to numerical columns.
    This function separates categorical and numerical columns from the input Polars DataFrames, applies one-hot encoding
    to the categorical columns, and standard scaling to the numerical columns. The transformed columns are then concatenated
    back together to form the final processed DataFrames for both training and validation sets.
    :param X_train: The training dataset containing both categorical and numerical columns.
    :type X_train: pl.DataFrame
    :param X_val: The validation dataset containing both categorical and numerical columns.
    :type X_val: pl.DataFrame
    :returns:
        - X_train_final: The processed training DataFrame, with categorical columns one-hot encoded
        and numerical columns scaled.
        - X_test_final: The processed validation DataFrame, with columns one-hot encoded
        and numerical columns scaled.
    :rtype: tuple[pl.DataFrame, pl.DataFrame]
    """

    logging.basicConfig(filename='train_and_evaluate_model.log', format='%(asctime)s %(message)s', level=logging.INFO)

    categorical_column = [col for col, dtype in zip(X_train.columns, X_train.dtypes) if dtype == pl.String]
    numerical_columns = [col for col, dtype in zip(X_train.columns, X_train.dtypes) if dtype in [pl.Int64, pl.Float32]]

    logger.info("columnas numéricas y categóricas separadas")
    
    X_train_cat = X_train.select(categorical_column).to_pandas() # Para no perder el nombre de las columnas
    logger.info('X_train_cat obtenido')
    X_val_cat = X_val.select(categorical_column).to_pandas()
    logger.info('X_test_cat obtenido')

    X_train_num = X_train.select(numerical_columns).to_pandas()
    logger.info('X_train_num obtenido')
    X_val_num = X_val.select(numerical_columns).to_pandas()
    logger.info('X_test_num obtenido')

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    logger.info('Instancia de OneHotEncoder creada')
    se = StandardScaler()
    logger.info('Insancia de StandardScaler creada')
    
    X_train_cat_ohe = ohe.fit_transform(X_train_cat)
    logger.info('onehotencoder aplicado a X_train_cat')
    X_val_cat_ohe = ohe.transform(X_val_cat)
    logger.info('onehotencoder aplicado a X_test_cat')

    X_train_num_scaled = se.fit_transform(X_train_num)
    logger.info('standardscaler aplicado a X_train_num')
    X_val_num_scaled = se.transform(X_val_num)
    logger.info('standardscaler aplicado a X_test_num')

    X_train_cat_ohe_df = pl.DataFrame(data=X_train_cat_ohe, schema=ohe.get_feature_names_out(categorical_column).tolist())
    logger.info('X_train_cat_ohe_df obtenido')
    X_train_num_scaled_df = pl.DataFrame(data=X_train_num_scaled, schema=numerical_columns)
    logger.info('X_train_num_scaled_df obtenido')
    X_train_final = pl.concat(items=[X_train_cat_ohe_df, X_train_num_scaled_df], how='horizontal')
    logger.info('X_train_final obtenido')
    
    X_val_cat_ohe_df = pl.DataFrame(data=X_val_cat_ohe, schema=ohe.get_feature_names_out(categorical_column).tolist())
    logger.info('X_test_cat_ohe_df obtenido')
    X_val_num_scaled_df = pl.DataFrame(data=X_val_num_scaled, schema=numerical_columns)
    logger.info('X_test_num_scaled_df obtenido')
    X_val_final = pl.concat(items=[X_val_cat_ohe_df, X_val_num_scaled_df], how='horizontal')
    logger.info('X_test_final obtenido')

    return X_train_final, X_val_final

def normalize_and_scale_test_data(X_test: pl.DataFrame) -> pl.DataFrame:
    """
    Normalizes and scales the test data by applying one-hot encoding to categorical columns
    and standard scaling to numerical columns.
    This function separates the categorical and numerical columns from the input Polars DataFrame,
    applies one-hot encoding to the categorical columns, and standard scaling to the numerical columns.
    The transformed columns are then concatenated and returned as a new Polars DataFrame.
    :param X_test: The input test data as a Polars DataFrame. It should contain both categorical (string) and
        numerical (integer or float) columns.
    :type X_test: pl.DataFrame    
    :return: A Polars DataFrame with categorical columns one-hot encoded and numerical columns standardized.
    :rtype: pl.DataFrame
        
    """

    logging.basicConfig(filename='train_and_evaluate_model.log', format='%(asctime)s %(message)s', level=logging.INFO)

    categorical_column = [col for col, dtype in zip(X_test.columns, X_test.dtypes) if dtype == pl.String]
    numerical_columns = [col for col, dtype in zip(X_test.columns, X_test.dtypes) if dtype in [pl.Int64, pl.Float32]]

    logger.info("columnas numéricas y categóricas separadas")
    
    X_test_cat = X_test.select(categorical_column).to_pandas() # Para no perder el nombre de las columnas
    logger.info('X_test_cat obtenido')

    X_test_num = X_test.select(numerical_columns).to_pandas()
    logger.info('X_test_num obtenido')

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    logger.info('Instancia de OneHotEncoder creada')
    se = StandardScaler()
    logger.info('Insancia de StandardScaler creada')
    
    X_test_cat_ohe = ohe.fit_transform(X_test_cat)
    logger.info('onehotencoder aplicado a X_test_cat')

    X_test_num_scaled = se.fit_transform(X_test_num)
    logger.info('standardscaler aplicado a X_test_num')

    X_test_cat_ohe_df = pl.DataFrame(data=X_test_cat_ohe, schema=ohe.get_feature_names_out(categorical_column).tolist())
    logger.info('X_test_cat_ohe_df obtenido')
    X_test_num_scaled_df = pl.DataFrame(data=X_test_num_scaled, schema=numerical_columns)
    logger.info('X_test_num_scaled_df obtenido')
    X_test_final = pl.concat(items=[X_test_cat_ohe_df, X_test_num_scaled_df], how='horizontal')
    logger.info('X_train_final obtenido')

    return X_test_final