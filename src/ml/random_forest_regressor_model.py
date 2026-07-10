import polars as pl
import numpy as np
from minio import Minio
import onnxruntime as rt

from pathlib import Path
import json
import logging
import os
import io

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

from minio_client import (
    connect_to_minio,
    download_model_from_minio,
    store_model_at_minio,
)

from utils import (
    split_train_validation_and_test_data,
    split_train_and_validation_data,
    normalize_and_scale_train_and_val_data,
    normalize_and_scale_test_data,
)

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    X_train_final: pl.DataFrame,
    X_val_final: pl.DataFrame,
    y_train: pl.Series,
    y_val: pl.Series,
) -> tuple[
    RandomForestRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict
]:
    """
    Trains a Random Forest Regressor using randomized hyperparameter search and evaluates its performance on a validation set.
    This function converts the provided training and validation data from Polars DataFrames and Series to NumPy arrays,
    performs hyperparameter optimization using RandomizedSearchCV, fits the best model, evaluates it on the validation set,
    and saves the predictions and true values to a CSV file. It returns the trained model, processed data, best parameters, and evaluation metrics.
    :param X_train_final: The training features as a Polars DataFrame.
    :type X_train_final: pl.DataFrame
    :param X_val_final: The validation features as a Polars DataFrame.
    :type X_val_final: pl.DataFrame
    :param y_train: The training target values as a Polars Series.
    :type y_train: pl.Series
    :param y_val: The validation target values as a Polars Series.
    :type y_val: pl.Series
    :returns:
        - best_model: The trained Random Forest Regressor with the best found hyperparameters.
        - X_train: The training features as a NumPy array.
        - X_val: The validation features as a NumPy array.
        - y_train: The training target values as a NumPy array.
        - y_val: The validation target values as a NumPy array.
        - params_: The best hyperparameters found during randomized search.
        - model_metrics: Dictionary containing evaluation metrics: MAE, MSE, RMSE, and R2 score.
    :rtype: tuple[RandomForestRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]
    """

    X_train = (
        X_train_final.to_numpy()
    )  # El método fit de Random Forest Regressor de sklearn necesita que sea MatrixLike
    y_train = y_train.to_numpy()

    X_val = X_val_final.to_numpy()
    y_val = y_val.to_numpy()

    rfr = RandomForestRegressor(n_jobs=-1, random_state=42)

    logger.info("Instancia de RFR creada")

    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [12, 15, 20],
        "min_samples_split": [50, 100, 150],
        "min_samples_leaf": [50, 100, 150],
    }

    rfr_grid = RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_grid,
        n_iter=25,
        cv=4,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    logger.info("Iniciando entrenamiento")
    rfr_grid.fit(X_train, y_train)

    logger.info("Entrenamiento terminado")

    best_model = rfr_grid.best_estimator_

    BEST_MODEL_N_ESTIMATOR = rfr_grid.best_params_["n_estimators"]
    BEST_MODEL_MAX_DEPTH = rfr_grid.best_params_["max_depth"]
    BEST_MODEL_MIN_SAMPLES_LEAF = rfr_grid.best_params_["min_samples_leaf"]
    BEST_MODEL_MIN_SAMPLES_SPLIT = rfr_grid.best_params_["min_samples_split"]

    params_ = {
        "N_ESTIMATOR": BEST_MODEL_N_ESTIMATOR,
        "MAX_DEPTH": BEST_MODEL_MAX_DEPTH,
        "MIN_SAMPLES_LEAF": BEST_MODEL_MIN_SAMPLES_LEAF,
        "MIN_SAMPLES_SPLIT": BEST_MODEL_MIN_SAMPLES_SPLIT,
    }

    best_params_path = Path("datasets/metrics/Random Forest")
    best_params_file = Path("datasets/metrics/Random Forest/best_params.json")
    if not os.path.exists(best_params_path):
        os.makedirs(name=best_params_path, exist_ok=True)

    with open(file=best_params_file, mode="w") as f:
        json.dump(params_, f)

    logger.info("Empezando validación")
    y_pred = best_model.predict(X_val)
    logger.info("Validación terminada")
    logger.info("prediccion obtenida")

    RFR_MAE = np.round(mean_absolute_error(y_val, y_pred), 2)
    RFR_MSE = np.round(mean_squared_error(y_val, y_pred), 2)
    RFR_RMSE = np.round(root_mean_squared_error(y_val, y_pred), 2)
    RFR_R2 = np.round(r2_score(y_val, y_pred), 2)

    model_metrics = {
        "RFR_MAE": RFR_MAE,
        "RFR_MSE": RFR_MSE,
        "RFR_RMSE": RFR_RMSE,
        "RFR_R2": RFR_R2,
    }

    model_metrics_path = Path("datasets/metrics/Random Forest")
    model_metrics_file = Path("datasets/metrics/Random Forest/validation_metrics.json")
    if not os.path.exists(model_metrics_path):
        os.makedirs(name=model_metrics_path, exist_ok=True)

    with open(file=model_metrics_file, mode="w") as f:
        json.dump(model_metrics, f)

    ys_train_and_val_path = Path(
        "../../../plot-streaming-data/src/plot_streaming_data/data/RandomForestRegressor/val"
    )
    if not os.path.exists(ys_train_and_val_path):
        os.makedirs(ys_train_and_val_path, exist_ok=True)

        ys_data = [
            pl.Series(name="id", values=np.arange(1, len(y_pred) + 1), dtype=pl.Int32),
            pl.Series("y_val", values=y_val, dtype=pl.Float32),
            pl.Series(name="y_pred", values=y_pred, dtype=pl.Float32),
        ]

        ys_df = pl.DataFrame(data=ys_data)
        ys_df.write_csv(file=f"{ys_train_and_val_path}/ys_val.csv")

    return best_model, X_train, X_val, y_train, y_val, params_, model_metrics


def test_model(
    best_model: rt.InferenceSession,
    X_test_final: pl.DataFrame,
    y_test: pl.Series,
    minio_client: Minio,
) -> np.ndarray:
    """
    Tests a trained ONNX model on the provided test dataset, computes evaluation metrics, saves metrics to MinIO,
    and optionally writes predictions and true values to a CSV file.
    :param best_model: The ONNX runtime inference session for the trained model.
    :type best_model: rt.InferenceSession
    :param X_test_final: The test features as a Polars DataFrame.
    :type X_test_final: pl.DataFrame
    :param y_test: The true target values for the test set as a Polars Series.
    :type y_test: pl.Series
    :param minio_client: The MinIO client used to upload the model metrics.
    :type minio_client: Minio
    :return: The predicted values for the test set.
    :rtype: np.ndarray
    """

    y_test = y_test.to_numpy()
    logger.info("y_val pasado a ndarray")

    X_test = X_test_final.to_numpy().astype(np.float32)

    input_name = best_model.get_inputs()[0].name

    y_pred = best_model.run(None, {input_name: X_test})[0]
    logger.info("Validación terminada")
    logger.info("prediccion obtenida")

    RFR_MAE = np.round(mean_absolute_error(y_test, y_pred), 2)
    RFR_MSE = np.round(mean_squared_error(y_test, y_pred), 2)
    RFR_RMSE = np.round(root_mean_squared_error(y_test, y_pred), 2)
    RFR_R2 = np.round(r2_score(y_test, y_pred), 2)

    data = {
        "RFR_MAE": RFR_MAE,
        "RFR_MSE": RFR_MSE,
        "RFR_RMSE": RFR_RMSE,
        "RFR_R2": RFR_R2,
    }

    bucket = os.getenv("RFR_MINIO_BUCKET")

    serialized_model_metrics = json.dumps(data).encode("utf-8")
    minio_client.put_object(
        bucket_name=bucket,
        object_name="/test/model_metrics.json",
        data=io.BytesIO(serialized_model_metrics),
        length=len(serialized_model_metrics),
        content_type="application/json",
    )

    ys_test_path = Path(
        "../../../plot-streaming-data/src/plot_streaming_data/data/RandomForestRegressor/test"
    )
    if not os.path.exists(ys_test_path):
        os.makedirs(ys_test_path, exist_ok=True)

        ys_data = [
            pl.Series(name="id", values=np.arange(1, len(y_pred) + 1), dtype=pl.Int32),
            pl.Series(name="y_test", values=y_test),
            pl.Series(name="y_pred", values=y_pred.ravel()),
        ]

        ys_df = pl.DataFrame(data=ys_data)
        ys_df.write_csv(file=f"{ys_test_path}/ys_test.csv")

    return y_pred


def pipeline_rfr():

    OBJECTS_CONTAINED_IN_BUCKET = 8

    df = pl.read_parquet(source=Path("datasets/final_data.parquet"))

    minio_client = connect_to_minio()

    bucket = os.environ.get("RFR_MINIO_BUCKET")
    objects = minio_client.list_objects(bucket_name=bucket, recursive=True)
    object_list = list(objects)

    if (minio_client.bucket_exists(bucket_name=bucket)) and (
        len(object_list) == OBJECTS_CONTAINED_IN_BUCKET
    ):
        test_data, train_validation_data, X_test, y_test = (
            split_train_validation_and_test_data(
                df=df,
                train_validation_data_path=Path(
                    "datasets/train_validation_data.parquet"
                ),
                test_data_path=Path("datasets/test_data.parquet"),
            )
        )
        X_test_final = normalize_and_scale_test_data(X_test=X_test)

        infsess = download_model_from_minio(model="rfr", minio_client=minio_client)

        test_model(
            best_model=infsess,
            X_test_final=X_test_final,
            y_test=y_test,
            minio_client=minio_client,
        )
    else:
        test_data, train_validation_data, X_test, y_test = (
            split_train_validation_and_test_data(
                df=df,
                train_validation_data_path=Path(
                    "datasets/train_validation_data.parquet"
                ),
                test_data_path=Path("datasets/test_data.parquet"),
            )
        )

        X_train, X_val, y_train, y_val = split_train_and_validation_data(
            df=train_validation_data
        )

        X_train_final, X_val_final = normalize_and_scale_train_and_val_data(
            X_train=X_train, X_val=X_val
        )

        rfr, X_train, X_val, y_train, y_val, params_, model_metrics = (
            train_and_evaluate_model(
                X_train_final=X_train_final,
                X_val_final=X_val_final,
                y_train=y_train,
                y_val=y_val,
            )
        )

        store_model_at_minio(
            model=rfr,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            minio_client=minio_client,
            params_=params_,
            model_metrics=model_metrics,
        )

        X_test_final = normalize_and_scale_test_data(X_test=X_test)

        infsess = download_model_from_minio(model="rfr", minio_client=minio_client)

        y_pred = test_model(
            best_model=infsess,
            X_test_final=X_test_final,
            y_test=y_test,
            minio_client=minio_client,
        )

    return y_pred


if __name__ == "__main__":
    pipeline_rfr()
