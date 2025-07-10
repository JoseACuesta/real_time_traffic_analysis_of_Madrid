import polars as pl
import numpy as np
from minio import Minio, error
from skl2onnx import to_onnx
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType

from pathlib import Path
import json
import logging
import os
import io

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    root_mean_squared_error,
    r2_score
)

from utils import (
    connect_to_minio,
    split_train_validation_and_test_data,
    split_train_and_validation_data,
    normalize_and_scale_train_and_val_data,
    normalize_and_scale_test_data
)

logger = logging.getLogger(__name__)

def train_and_evaluate_model(
    X_train_final: pl.DataFrame, X_val_final: pl.DataFrame, y_train: pl.Series, y_val: pl.Series
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:

    X_train = X_train_final.to_numpy() # El método fit de Random Forest Regressor de sklearn necesita que sea MatrixLike
    y_train = y_train.to_numpy()

    X_val = X_val_final.to_numpy()
    y_val = y_val.to_numpy()

    rfr = RandomForestRegressor(n_jobs=-1, random_state=42)
    
    logger.info('Instancia de RFR creada')
    
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [12,15,20],
        'min_samples_split': [50,100,150],
        'min_samples_leaf': [50,100,150]
    }

    rfr_grid = RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_grid,
        n_iter=25,
        cv=4,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    logger.info('Iniciando entrenamiento')
    rfr_grid.fit(X_train, y_train)

    logger.info('Entrenamiento terminado')

    best_model = rfr_grid.best_estimator_

    BEST_MODEL_N_ESTIMATOR = rfr_grid.best_params_['n_estimators']
    BEST_MODEL_MAX_DEPTH = rfr_grid.best_params_['max_depth']
    BEST_MODEL_MIN_SAMPLES_LEAF = rfr_grid.best_params_['min_samples_leaf']
    BEST_MODEL_MIN_SAMPLES_SPLIT = rfr_grid.best_params_['min_samples_split']

    params_ = {
        'N_ESTIMATOR': BEST_MODEL_N_ESTIMATOR,
        'MAX_DEPTH': BEST_MODEL_MAX_DEPTH,
        'MIN_SAMPLES_LEAF': BEST_MODEL_MIN_SAMPLES_LEAF,
        'MIN_SAMPLES_SPLIT': BEST_MODEL_MIN_SAMPLES_SPLIT
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

    ys_train_and_val_path = Path('../../../plot-streaming-data/src/plot_streaming_data/data/val')
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

def store_model(
    model: RandomForestRegressor,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    params_: dict,
    model_metrics: dict,
    minio_client: Minio
) -> None:
    
    bucket = os.getenv('RFR_MINIO_BUCKET')
    if not minio_client.bucket_exists(bucket_name=bucket):
        minio_client.make_bucket(bucket_name=bucket)

    initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]
    onnx = to_onnx(model=model, initial_types=initial_type)
    model_bytes = onnx.SerializeToString()
    model_buffer = io.BytesIO(model_bytes)

    MODEL_ID = 'RandomForestRegressor'

    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/model.onnx',
        data=model_buffer,
        length=len(model_bytes),
        content_type="application/train_and_val/octet-stream"
    )

    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, X_train, delimiter=',')
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    csv_io = io.BytesIO(csv_bytes)
    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/X_train.csv',
        data=csv_io,
        length=len(csv_bytes),
        content_type='text/csv'
    )

    np.savetxt(csv_buffer, X_val, delimiter=',')
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    csv_io = io.BytesIO(csv_bytes)
    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/X_val.csv',
        data=csv_io,
        length=len(csv_bytes),
        content_type='text/csv'
    )

    np.savetxt(csv_buffer, y_train, delimiter=',')
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    csv_io = io.BytesIO(csv_bytes)
    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/y_train.csv',
        data=csv_io,
        length=len(csv_bytes),
        content_type='text/csv'
    )

    np.savetxt(csv_buffer, y_val, delimiter=',')
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    csv_io = io.BytesIO(csv_bytes)
    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/y_val.csv',
        data=csv_io,
        length=len(csv_bytes),
        content_type='text/csv'
    )

    serialized_model_params = json.dumps(params_).encode('utf-8')
    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/model_params.json',
        data=io.BytesIO(serialized_model_params),
        length=len(serialized_model_params),
        content_type='application/json'
    )

    serialized_model_metrics = json.dumps(model_metrics).encode('utf-8')
    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/model_metrics.json',
        data=io.BytesIO(serialized_model_metrics),
        length=len(serialized_model_metrics),
        content_type='application/json'
    )
    
def download_model_from_minio(minio_client: Minio) -> rt.InferenceSession:
    
    bucket = os.getenv('RFR_MINIO_BUCKET')
    MODEL_ID = 'RandomForestRegressor'

    try:
        onnx_model = minio_client.get_object(bucket_name=bucket, object_name=f'{MODEL_ID}/train_and_val/model.onnx').data
    except error.S3Error:
        raise ValueError(f'Model not found for {MODEL_ID}/train_and_val/model.onnx')
    infses = rt.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
    
    return infses

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

    MODEL_ID = 'RandomForestRegressor'
    bucket = os.getenv('RFR_MINIO_BUCKET')

    serialized_model_metrics = json.dumps(data).encode('utf-8')
    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/test/model_metrics.json',
        data=io.BytesIO(serialized_model_metrics),
        length=len(serialized_model_metrics),
        content_type='application/json'
    )

    ys_test_path = Path('../../../plot-streaming-data/src/plot_streaming_data/data/test')
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

    TRAIN_AND_VAL_MINIO_OBJECTS = 1

    list_objects = []
    bucket = os.getenv('RFR_MINIO_BUCKET')
    objects = minio_client.list_objects(bucket_name=bucket, recursive=True)
    for obj in objects: 
        list_objects.append(obj)

    if len(list_objects) >= TRAIN_AND_VAL_MINIO_OBJECTS:
        test_data, train_validation_data, X_test, y_test = split_train_validation_and_test_data(
        df=df,
        train_validation_data_path=Path('data/train_validation_data.parquet'),
        test_data_path=Path('data/test_data.parquet')
    )
        X_test_final = normalize_and_scale_test_data(X_test=X_test)

        infsess = download_model_from_minio(minio_client=minio_client)

        test_model(
            best_model=infsess,
            X_test_final = X_test_final,
            y_test=y_test,
            minio_client=minio_client)
    else:
        test_data, train_validation_data, X_test, y_test = split_train_validation_and_test_data(
            df=df,
            train_validation_data_path=Path('data/train_validation_data.parquet'),
            test_data_path=Path('data/test_data.parquet')
        )

        X_train, X_val, y_train, y_val = split_train_and_validation_data(df=train_validation_data)

        X_train_final, X_val_final = normalize_and_scale_train_and_val_data(X_train=X_train, X_val=X_val)

        rfr, X_train, X_val, y_train, y_val, params_, model_metrics = train_and_evaluate_model(
            X_train_final=X_train_final,
            X_val_final=X_val_final,
            y_train=y_train,
            y_val=y_val)
        
        store_model(
            model=rfr,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            minio_client=minio_client,
            params_=params_,
            model_metrics=model_metrics
        )
        
        X_test_final = normalize_and_scale_test_data(X_test=X_test)

        infsess = download_model_from_minio(minio_client=minio_client)

        y_pred = test_model(
            best_model=infsess,
            X_test_final = X_test_final,
            y_test=y_test,
            minio_client=minio_client)
        
        print(y_pred)
    
if __name__ == "__main__":
    main()
    