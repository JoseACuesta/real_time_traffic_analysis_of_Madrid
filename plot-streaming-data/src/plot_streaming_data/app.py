import pandas as pd

from pathlib import Path
import os
import threading
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

import streamlit as st
import plotly.graph_objects as go

def split_csv_file_into_streaming_directory(file_path: Path, streaming_directory: Path):
    """
    Splits a CSV file into multiple single-row CSV files in a specified directory.
    :param file_path: Path to the input CSV file to be split.
    :type file_path: Path
    :param streaming_directory: Directory where the split CSV files will be saved.
    :type streaming_directory: Path
    """
    
    MAX_FILES = 5000
    with open(file=file_path, mode='r') as file_:
        os.makedirs(name=streaming_directory, exist_ok=True)
        header = file_.readline()
        for i, line in enumerate(file_):
            if (i <= MAX_FILES):
                path = f'{streaming_directory}/ys_{i:06d}.csv'
                with open(file=path, mode='w') as f:
                    f.write(header)
                    f.write(line)

def initialise_streaming():
    """
    Initializes a Spark session to simulate streaming data processing.
    This function sets up a local Spark session, defines a schema for the input data,
    reads CSV files in streaming mode from the 'data/file_sink/' directory, and writes
    the results to the console in continuous update mode.
    :raises pyspark.sql.utils.StreamingQueryException: If an error occurs during streaming execution.
    """

    spark = SparkSession.builder \
        .appName("Simulate Streaming") \
        .master("local[8]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("y_test", FloatType(), True),
        StructField("y_pred", FloatType(), True)
    ])

    df = spark.readStream \
        .option("header", True) \
        .schema(schema) \
        .csv("data/file_sink/")
    
    query = df \
        .writeStream \
        .outputMode('update') \
        .format('console') \
        .start()
    
    query.awaitTermination()

def plot_data_plotly():
    """
    Visualizes in real time the comparison between predicted and actual traffic values using Plotly and Streamlit.
    This function iterates over CSV files generated in real time, extracts prediction and actual values, and displays them in an interactive plot.
    It detects anomalies when the difference between the predicted and actual value exceeds a threshold (20% of the actual value), highlighting them in the plot and showing a warning in the interface.
    Processed files are deleted after visualization.
    :raises Exception: Displays an error message in Streamlit if any issue occurs while processing the files.
    """

    st.title("Predicción vs Real - Streaming")

    SLEEP = 1

    x_vals_list = []
    y_pred_list = []
    y_test_list = []
    y_pred_colors = []

    total_frames = len([f for f in os.listdir('data/file_sink') if f.endswith('.csv')])

    plotly_chart = st.empty()

    for i in range(total_frames):
        try:
            path = f'data/file_sink/ys_{i:06d}.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                x_vals = int(df.loc[0, 'id'])
                y_pred_value = float(df.loc[0, 'y_pred'])
                y_test_value = float(df.loc[0, 'y_test'])

                x_vals_list.append(x_vals)
                y_pred_list.append(y_pred_value)
                y_test_list.append(y_test_value)
                
                THRESHOLD = 0.2 * y_test_value
                
                is_anomaly = abs(y_pred_value - y_test_value) > THRESHOLD
                if is_anomaly:
                    st.warning(body=f'ANOMALY: The value of the load of traffic was supossed to be: {y_test_value}, but {y_pred_value} has been predicted', icon='⚠️')
                    y_pred_colors.append('red')
                else:
                    y_pred_colors.append('blue')

                fig = go.Figure()
                fig.add_traces([
                    go.Scatter(x=x_vals_list, y=y_pred_list, mode='lines+markers', name='y_pred', line=dict(color='blue'), marker=dict(color=y_pred_colors)),
                    go.Scatter(x=x_vals_list, y=y_test_list, mode='lines+markers', name='y_test', line=dict(color='green'))
                ])

                fig.update_layout(
                xaxis_title="Muestra",
                yaxis_title="Valor",
                title="Predicción vs Real (Streaming)",
                showlegend=True
                )

                plotly_chart.plotly_chart(fig, use_container_width=True, key=f'dinamyc_plot_{i}')
                os.remove(path)
                time.sleep(SLEEP)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            continue

def main():
    split_csv_file_into_streaming_directory(
        file_path=Path('data/test/ys_test.csv'),
        streaming_directory=Path('data/file_sink')
    )

    threading.Thread(target=initialise_streaming, daemon=True).start()
    plot_data_plotly()

if __name__ == "__main__":
    main()
