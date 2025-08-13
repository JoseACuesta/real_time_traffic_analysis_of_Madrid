from minio import Minio, error
from dotenv import load_dotenv
import os
import io
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from skl2onnx import to_onnx
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType

def connect_to_minio() -> Minio:
    """
    Establishes a connection to a MinIO object storage server using credentials loaded from environment variables.
    The function loads environment variables from a .env file and retrieves the MinIO service endpoint, access key, and secret key.
    It then creates and returns a Minio client instance configured with these credentials.
    :returns: An instance of the Minio client connected to the specified MinIO service.
    :rtype: Minio
    :raises KeyError: If any of the required environment variables ('MINIO_SERVICE', 'MINIO_ACCESS_KEY', 'MINIO_SECRET_KEY') are not set.
    """

    load_dotenv()

    minio_service = os.environ.get('MINIO_SERVICE')
    minio_access_key = os.environ.get('MINIO_ACCESS_KEY')
    minio_secret_key = os.environ.get('MINIO_SECRET_KEY')

    client = Minio(
        minio_service,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    
    return client

def store_model_at_minio( 
    model: RandomForestRegressor | XGBRegressor,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    params_: dict,
    model_metrics: dict,
    minio_client: Minio
) -> None:
    """
    Stores a trained machine learning model, its training/validation data, parameters, and metrics in a MinIO bucket.
    The function serializes the given model to ONNX format and uploads it to the specified MinIO bucket. It also uploads
    the training and validation datasets (features and targets) as CSV files, and the model's parameters and metrics as JSON files.
    If the specified bucket does not exist, it will be created.
    :param model: Trained machine learning model (RandomForestRegressor or XGBRegressor) to be stored.
    :type model: RandomForestRegressor | XGBRegressor
    :param X_train: Training feature data.
    :type X_train: np.ndarray
    :param X_val: Validation feature data.
    :type X_val: np.ndarray
    :param y_train: Training target data.
    :type y_train: np.ndarray
    :param y_val: Validation target data.
    :type y_val: np.ndarray
    :param params_: Model hyperparameters and configuration.
    :type params_: dict
    :param model_metrics: Dictionary containing model evaluation metrics.
    :type model_metrics: dict
    :param minio_client: Initialized MinIO client for object storage operations.
    :type minio_client: Minio
    :raises ValueError: If the MINIO_BUCKET environment variable is not set.
    :raises Exception: If there is an error during serialization or upload to MinIO.
    :return: None
    """
    
    bucket = os.environ.get('MINIO_BUCKET')
    if not minio_client.bucket_exists(bucket_name=bucket):
        minio_client.make_bucket(bucket_name=bucket)

    initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]
    onnx = to_onnx(model=model, initial_types=initial_type)
    model_bytes = onnx.SerializeToString()
    model_buffer = io.BytesIO(model_bytes)

    MODEL_ID = 'RandomForestRegressor' if type(model) == RandomForestRegressor else 'XGBRegressor'

    minio_client.put_object(
        bucket_name=bucket,
        object_name=f'{MODEL_ID}/train_and_val/model.onnx',
        data=model_buffer,
        length=len(model_bytes),
        content_type="application/train_and_val/octet-stream"
    )

    csv_buffer = io.StringIO() # A utilizar por todos los .csv

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
    """
    Downloads an ONNX model from a MinIO bucket and loads it into an ONNX Runtime InferenceSession.
    :param minio_client: An instance of the Minio client used to interact with the MinIO object storage.
    :type minio_client: Minio
    :return: An ONNX Runtime InferenceSession initialized with the downloaded model.
    :rtype: rt.InferenceSession
    :raises ValueError: If the specified model file is not found in the MinIO bucket.
    """
    
    bucket = os.environ.get('RFR_MINIO_BUCKET')
    MODEL_ID = 'RandomForestRegressor'

    try:
        onnx_model = minio_client.get_object(bucket_name=bucket, object_name=f'{MODEL_ID}/train_and_val/model.onnx').data
    except error.S3Error:
        raise ValueError(f'Model not found for {MODEL_ID}/train_and_val/model.onnx')
    infses = rt.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
    
    return infses