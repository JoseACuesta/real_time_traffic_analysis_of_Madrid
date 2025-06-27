import polars as pl
#import pandas as pd
import numpy as np

from pathlib import Path
import json
import joblib
import logging

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    root_mean_squared_error,
    r2_score
)

# from dask.distributed import Client, LocalCluster

from transform_data import raw_data_from_polars_dataframe, transform_polars_dataframe

logger = logging.getLogger(__name__)

def split_train_and_test_data(df: pl.DataFrame):
    
    is_2023 = df['year'] == 2023.0

    X = df.drop('carga')
    y = df['carga']

    X_train = X.filter(~is_2023)
    X_test = X.filter(is_2023)

    y_train = y.filter(~is_2023)
    y_test = y.filter(is_2023)

    return X_train, X_test, y_train, y_test

def normalize_and_scale_data_polars(X_train: pl.DataFrame, X_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    
    logging.basicConfig(filename='train_and_evaluate_model.log', level=logging.INFO)

    categorical_column = [col for col, dtype in zip(X_train.columns, X_train.dtypes) if dtype == pl.String]
    numerical_columns = [col for col, dtype in zip(X_train.columns, X_train.dtypes) if dtype in [pl.Int64, pl.Float32]]

    logger.info("columnas numéricas y categóricas separadas")
    
    X_train_cat = X_train.select(categorical_column).to_pandas() # Para no perder el nombre de las columnas
    logger.info('X_train_cat obtenido')
    X_test_cat = X_test.select(categorical_column).to_pandas()
    logger.info('X_test_cat obtenido')

    X_train_num = X_train.select(numerical_columns).to_pandas()
    logger.info('X_train_num obtenido')
    X_test_num = X_test.select(numerical_columns).to_pandas()
    logger.info('X_test_num obtenido')

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    logger.info('Instancia de OneHotEncoder creada')
    se = StandardScaler()
    logger.info('Insancia de StandardScaler creada')
    
    X_train_cat_ohe = ohe.fit_transform(X_train_cat)
    logger.info('onehotencoder aplicado a X_train_cat')
    X_test_cat_ohe = ohe.transform(X_test_cat)
    logger.info('onehotencoder aplicado a X_test_cat')

    X_train_num_scaled = se.fit_transform(X_train_num)
    logger.info('standardscaler aplicado a X_train_num')
    X_test_num_scaled = se.transform(X_test_num)
    logger.info('standardscaler aplicado a X_test_num')

    X_train_cat_ohe_df = pl.DataFrame(data=X_train_cat_ohe, schema=ohe.get_feature_names_out(categorical_column).tolist())
    logger.info('X_train_cat_ohe_df obtenido')
    X_train_num_scaled_df = pl.DataFrame(data=X_train_num_scaled, schema=numerical_columns)
    logger.info('X_train_num_scaled_df obtenido')
    X_train_final = pl.concat(items=[X_train_cat_ohe_df, X_train_num_scaled_df], how='horizontal')
    logger.info('X_train_final obtenido')
    
    X_test_cat_ohe_df = pl.DataFrame(data=X_test_cat_ohe, schema=ohe.get_feature_names_out(categorical_column).tolist())
    logger.info('X_test_cat_ohe_df obtenido')
    X_test_num_scaled_df = pl.DataFrame(data=X_test_num_scaled, schema=numerical_columns)
    logger.info('X_test_num_scaled_df obtenido')
    X_test_final = pl.concat(items=[X_test_cat_ohe_df, X_test_num_scaled_df], how='horizontal')
    logger.info('X_test_final obtenido')

    return X_train_final, X_test_final

def train_model(X_train_final: pl.DataFrame, y_train: pl.Series) -> RandomForestRegressor:

    X_train = X_train_final.slice(0, 1_000_000).to_numpy() # El método fit de Random Forest Regressor de sklearn necesita que sea MatrixLike
    y_train = y_train.slice(0, 1_000_000).to_numpy()
    logger.info('y_train pasado a ndarray')

    # local_cluster = LocalCluster(n_workers=4,
    #                              threads_per_worker=1,
    #                              memory_limit='3GiB',
    #                              dashboard_address='8787')
    
    #client = Client(local_cluster)

    rfr = RandomForestRegressor(n_jobs=-1)

    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [20, 30]
    }
    
    logger.info('Instancia de RFR creada')
    grid = RandomizedSearchCV(param_distributions=param_grid,
                              estimator=rfr,
                              n_iter=8,
                              cv=3,
                              scoring='neg_mean_absolute_error',
                              n_jobs=-1,
                              random_state=42)
    
    logger.info('Iniciando entrenamiento')

    with joblib.parallel_backend('dask'):
        grid.fit(X_train, y_train)

    logger.info('Entrenamiento terminado')
    best_model = grid.best_estimator_
    
    return best_model

def test_model(best_model: RandomForestRegressor, X_test_final: pl.DataFrame, y_test: pl.Series):
    
    y_test = y_test.slice(0, 200_000).to_numpy()
    logger.info('y_test pasado a ndarray')

    X_test = X_test_final.slice(0, 200_000).to_numpy()
    y_pred = best_model.predict(X_test)
    logger.info('prediccion obtenida')

    RFR_MAE = np.round(mean_absolute_error(y_test, y_pred), 2)
    RFR_MSE = np.round(mean_squared_error(y_test, y_pred), 2)
    RFR_RMSE = np.round(root_mean_squared_error(y_test, y_pred), 2)
    RFR_R2 = np.round(r2_score(y_test, y_pred), 2)

    data = {
        'RFR_MAE': RFR_MAE,
        'RFR_MSE': RFR_MSE,
        'RFR_RMSE': RFR_RMSE,
        'RFR_R2': RFR_R2
    }

    with open('data/metrics.json', mode='w') as metrics:
        json.dump(data, metrics)

def main():
    df = raw_data_from_polars_dataframe(path=Path('../../../data-preprocessing/src/data_preprocessing/data/provisional_final_data.csv'))
    df_transformed = transform_polars_dataframe(df)
    X_train, X_test, y_train, y_test = split_train_and_test_data(df_transformed)
    X_train_final, X_test_final = normalize_and_scale_data_polars(X_train, X_test)
    best_model = train_model(X_train_final, y_train)
    test_model(best_model, X_test_final, y_test)
    

if __name__ == "__main__":
    main()