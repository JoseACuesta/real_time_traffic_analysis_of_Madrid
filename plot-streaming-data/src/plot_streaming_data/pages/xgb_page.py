import streamlit as st
from pathlib import Path
import threading

from plot_streaming_data.utils.split_csv_files import split_xgboost_prediction_data_file_into_streaming_directory

from plot_streaming_data.utils.spark_client import initialise_xgboost_streaming

from plot_streaming_data.utils.plot_real_time_data import plot_xgboost_data_plotly

st.title('XGBoost Regressor Prediction')

split_xgboost_prediction_data_file_into_streaming_directory(
    file_path=Path('data/XGBoost/test/ys_test.csv'),
    streaming_directory=Path('data/XGBoost/file_sink')
)

threading.Thread(
    target=initialise_xgboost_streaming, daemon=True
).start()

plot_xgboost_data_plotly()