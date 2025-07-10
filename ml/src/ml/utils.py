import polars as pl
from minio import Minio
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import os
from dotenv import load_dotenv
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def connect_to_minio() -> Minio:

    load_dotenv()

    minio_service = os.getenv('MINIO_SERVICE')
    minio_access_key = os.getenv('MINIO_ACCESS_KEY')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY')

    client = Minio(
        minio_service,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    
    return client

def split_train_validation_and_test_data(df:pl.DataFrame, train_validation_data_path:Path, test_data_path:Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series]:
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
    
    is_2023 = df['year'] == 2023

    X = df.drop('carga')
    y = df['carga']

    X_train = X.filter(~is_2023)
    X_val = X.filter(is_2023)

    y_train = y.filter(~is_2023)
    y_val = y.filter(is_2023)

    return X_train, X_val, y_train, y_val

def normalize_and_scale_train_and_val_data(X_train: pl.DataFrame, X_val: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:

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