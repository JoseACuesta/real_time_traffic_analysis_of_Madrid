import streamlit as st
from pathlib import Path
import threading

from utils.split_csv_files import split_rfr_prediction_data_file_into_streaming_directory

from utils.spark_client import initialise_rfr_streaming

from utils.plot_real_time_data import plot_rfr_data_plotly

st.title('Random Forest Regressor Prediction')

split_rfr_prediction_data_file_into_streaming_directory(
    file_path=Path('data/RandomForestRegressor/test/ys_test.csv'),
    streaming_directory=Path('data/RandomForestRegressor/file_sink')
)

threading.Thread(
    target=initialise_rfr_streaming, daemon=True
).start()

plot_rfr_data_plotly()