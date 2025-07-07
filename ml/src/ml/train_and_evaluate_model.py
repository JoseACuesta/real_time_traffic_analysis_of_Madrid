import polars as pl
import numpy as np

from pathlib import Path
import json
import logging
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    root_mean_squared_error,
    r2_score
)

logger = logging.getLogger(__name__)

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

def train_and_evaluate_model(X_train_final: pl.DataFrame, X_val_final: pl.DataFrame, y_train: pl.Series, y_val: pl.Series) -> RandomForestRegressor:

    X_train = X_train_final.to_numpy() # El método fit de Random Forest Regressor de sklearn necesita que sea MatrixLike
    y_train = y_train.to_numpy()

    X_val = X_val_final.to_numpy()
    y_val = y_val.to_numpy()

    rfr = RandomForestRegressor(n_jobs=-1, random_state=42)
    
    logger.info('Instancia de RFR creada')
    
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [12,15,20],
        'min_samples_leaf': np.linspace(0.1, 1, 10)
    }

    rfr_grid = GridSearchCV(
        estimator=rfr,
        param_grid=param_grid,
        cv=4,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )

    logger.info('Iniciando entrenamiento')
    rfr_grid.fit(X_train, y_train)

    logger.info('Entrenamiento terminado')

    best_model = rfr_grid.best_estimator_

    BEST_MODEL_N_ESTIMATOR = rfr_grid.best_params_['n_estimators']
    BEST_MODEL_MAX_DEPTH = rfr_grid.best_params_['max_depth']
    BEST_MODEL_MIN_SAMPLES_LEAF = rfr_grid.best_params_['min_samples_leaf']

    params_ = {
        'N_ESTIMATOR': BEST_MODEL_N_ESTIMATOR,
        'MAX_DEPTH': BEST_MODEL_MAX_DEPTH,
        'MIN_SAMPLES_LEAF': BEST_MODEL_MIN_SAMPLES_LEAF
    }

    with open('data/metrics/model_params.json', mode='w') as params:
        json.dump(params_, params)
    
    logger.info('Empezando validación')
    y_pred = best_model.predict(X_val)
    logger.info('Validación terminada')
    logger.info('prediccion obtenida')

    RFR_MAE = np.round(mean_absolute_error(y_val, y_pred), 2)
    RFR_MSE = np.round(mean_squared_error(y_val, y_pred), 2)
    RFR_RMSE = np.round(root_mean_squared_error(y_val, y_pred), 2)
    RFR_R2 = np.round(r2_score(y_val, y_pred), 2)

    data = {
        'RFR_MAE': RFR_MAE,
        'RFR_MSE': RFR_MSE,
        'RFR_RMSE': RFR_RMSE,
        'RFR_R2': RFR_R2
    }

    with open('data/metrics/val_metrics.json', mode='w') as metrics:
        json.dump(data, metrics)

    with open('data/metrics/val_metrics.json', mode='r') as metrics:
        m = json.load(metrics)
        print(m)
    
    return best_model

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

def test_model(best_model: RandomForestRegressor, X_test_final: pl.DataFrame, y_test: pl.Series):
    
    y_test = y_test.to_numpy()
    logger.info('y_val pasado a ndarray')

    X_test = X_test_final.to_numpy()

    y_pred = best_model.predict(X_test)
    logger.info('Validación terminada')
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

    with open('data/metrics/test_metrics.json', mode='w') as metrics:
        json.dump(data, metrics)

def main():
    df = pl.read_parquet(source=Path('data/final_data.parquet'))

    print(df)
    print(df.shape)

    test_data, train_validation_data, X_test, y_test = split_train_validation_and_test_data(
        df=df,
        train_validation_data_path=Path('data/train_validation_data.parquet'),
        test_data_path=Path('data/test_data.parquet')
    )

    print("train_validation_data shape ", train_validation_data.shape)
    print("test_data shape: ", test_data.shape)
    print("X_test_shape: ", X_test.shape)
    print("y_test_shape: ", y_test.shape)

    X_train, X_val, y_train, y_val = split_train_and_validation_data(df=train_validation_data)

    print("X_train shape: ", X_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_train shape: ", y_train.shape)
    print("y_val shape: ", y_val.shape)

    X_train_final, X_val_final = normalize_and_scale_train_and_val_data(X_train=X_train, X_val=X_val)

    rfr = train_and_evaluate_model(
        X_train_final=X_train_final,
        X_val_final=X_val_final,
        y_train=y_train,
        y_val=y_val)
    
    X_test_final = normalize_and_scale_test_data(X_test=X_test)

    test_model(
        best_model=rfr,
        X_test_final = X_test_final,
        y_test=y_test)
    

if __name__ == "__main__":
    main()
    