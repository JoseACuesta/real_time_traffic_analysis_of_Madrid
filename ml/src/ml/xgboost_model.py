import polars as pl
import numpy as np
import onnxruntime as rt
from minio import Minio

import os
import io
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

from ml.minio_client import (
    connect_to_minio,
    download_model_from_minio,
    store_model_at_minio
)

from ml.utils import (
    split_train_validation_and_test_data,
    split_train_and_validation_data,
    normalize_and_scale_train_and_val_data,
    normalize_and_scale_test_data
)

logger = logging.getLogger(__name__)

def train_and_evaluate_model(
    X_train_final: pl.DataFrame, X_val_final: pl.DataFrame, y_train: pl.Series, y_val: pl.Series
) -> tuple[XGBRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    
    X_train = X_train_final.to_numpy()
    y_train = y_train.to_numpy()

    X_val = X_val_final.to_numpy()
    y_val = y_val.to_numpy()

    xgbreg = XGBRegressor()
    logger.info('Instancia de XGB creada')
    
    param_grid = {
        'objective': ['reg:squarederror'],
        'n_estimators': [100, 150],
        'max_depth': [4,5],
        'subsample': [0.7, 0.8],
        'learning_rate': [0.1, 0.01]
    }

    xgbreg_grid = GridSearchCV(
        estimator=xgbreg,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    logger.info('Iniciando entrenamiento')
    xgbreg_grid.fit(X_train, y_train)

    logger.info('Entrenamiento terminado')

    best_model = xgbreg_grid.best_estimator_

    BEST_MODEL_OBJECTIVE = xgbreg_grid.best_params_['objective']
    BEST_MODEL_N_ESTIMATORS = xgbreg_grid.best_params_['n_estimators']
    BEST_MODEL_MAX_DEPTH = xgbreg_grid.best_params_['max_depth']
    BEST_MODEL_SUBSAMPLE = xgbreg_grid.best_params_['subsample']
    BEST_MODEL_LEARNING_RATE = xgbreg_grid.best_params_['learning_rate']

    params_ = {
        'OBJECTIVE': BEST_MODEL_OBJECTIVE,
        'N_ESTIMATORS': BEST_MODEL_N_ESTIMATORS,
        'MAX_DEPTH': BEST_MODEL_MAX_DEPTH,
        'SUBSAMPLE': BEST_MODEL_SUBSAMPLE,
        'LEARNING_RATE': BEST_MODEL_LEARNING_RATE
    }

    logger.info('Empezando validación')
    y_pred = best_model.predict(X_val)
    logger.info('Validación terminada')
    logger.info('prediccion obtenida')

    RFR_MAE = np.round(mean_absolute_error(y_val, y_pred), 2)
    RFR_MSE = np.round(mean_squared_error(y_val, y_pred), 2)
    RFR_RMSE = np.round(root_mean_squared_error(y_val, y_pred), 2)
    RFR_R2 = np.round(r2_score(y_val, y_pred), 2)

    model_metrics = {
        'RFR_MAE': RFR_MAE,
        'RFR_MSE': RFR_MSE,
        'RFR_RMSE': RFR_RMSE,
        'RFR_R2': RFR_R2
    }

    ys_train_and_val_path = Path('../../../plot-streaming-data/src/plot_streaming_data/data/XGBoost/val')
    if not os.path.exists(ys_train_and_val_path):
        os.makedirs(ys_train_and_val_path, exist_ok=True)

        ys_data = [
            pl.Series(name='id', values=np.arange(1, len(y_pred) + 1), dtype=pl.Int32),
            pl.Series('y_val', values=y_val, dtype=pl.Float32),
            pl.Series(name='y_pred', values=y_pred, dtype=pl.Float32)
        ]

        ys_df = pl.DataFrame(data=ys_data)
        ys_df.write_csv(file=f'{ys_train_and_val_path}/ys_val.csv')
    
    return best_model, X_train, X_val, y_train, y_val, params_, model_metrics

def test_model(best_model: rt.InferenceSession, X_test_final: pl.DataFrame, y_test: pl.Series, minio_client: Minio) -> np.ndarray:
    
    y_test = y_test.to_numpy()
    logger.info('y_val pasado a ndarray')

    X_test = X_test_final.to_numpy().astype(np.float32)

    input_name = best_model.get_inputs()[0].name

    y_pred = best_model.run(None, {input_name: X_test})[0]

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

    bucket = os.getenv('XGBOOST_BUCKET')

    serialized_model_metrics = json.dumps(data).encode('utf-8')
    minio_client.put_object(
        bucket_name=bucket,
        object_name='/test/model_metrics.json',
        data=io.BytesIO(serialized_model_metrics),
        length=len(serialized_model_metrics),
        content_type='application/json'
    )

    ys_test_path = Path('../../../plot-streaming-data/src/plot_streaming_data/data/XGBoost/test')
    if not os.path.exists(ys_test_path):
        os.makedirs(ys_test_path, exist_ok=True)

        ys_data = [
            pl.Series(name='id', values=np.arange(1, len(y_pred) + 1), dtype=pl.Int32),
            pl.Series(name='y_test', values=y_test),
            pl.Series(name='y_pred', values=y_pred.ravel())
        ]

        ys_df = pl.DataFrame(data=ys_data)
        ys_df.write_csv(file=f'{ys_test_path}/ys_test.csv')
        
    return y_pred

def main():
    
    df = pl.read_parquet(source=Path('data/final_data.parquet'))

    minio_client = connect_to_minio()

    test_data, train_validation_data, X_test, y_test = split_train_validation_and_test_data(
            df=df,
            train_validation_data_path=Path('data/train_validation_data.parquet'),
            test_data_path=Path('data/test_data.parquet')
        )

    X_train, X_val, y_train, y_val = split_train_and_validation_data(df=train_validation_data)

    X_train_final, X_val_final = normalize_and_scale_train_and_val_data(X_train=X_train, X_val=X_val)

    xgb, X_train, X_val, y_train, y_val, params_, model_metrics = train_and_evaluate_model(
        X_train_final=X_train_final,
        X_val_final=X_val_final,
        y_train=y_train,
        y_val=y_val)
        
    X_test_final = normalize_and_scale_test_data(X_test=X_test)

    store_model_at_minio(
        model=xgb,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        minio_client=minio_client,
        params_=params_,
        model_metrics=model_metrics
        )
    
    infsess = download_model_from_minio(
        model = xgb,
        minio_client=minio_client
    )

    y_pred = test_model(
        best_model=infsess,
        X_test_final = X_test_final,
        y_test=y_test,
        minio_client=minio_client)
        
    print(y_pred)
    
if __name__ == "__main__":
    main()