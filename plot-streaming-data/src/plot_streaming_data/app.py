import numpy as np
import pandas as pd
import polars as pl

from pathlib import Path
import os
import threading
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

import streamlit as st
import plotly.graph_objects as go

def split_csv_file_into_streaming_directory(file_path: Path, streaming_directory: Path):
    with open(file=file_path, mode='r') as file_:
        os.makedirs(name=streaming_directory, exist_ok=True)
        header = file_.readline()
        for i, line in enumerate(file_):
            if (i != 0) or (len(line) > 1):
                path = f'{streaming_directory}/ys_{i:06d}.csv'
                with open(file=path, mode='w') as f:
                    f.write(header)
                    f.write(line)

def initialise_streaming():
        # 1. Crear sesión de Spark
    spark = SparkSession.builder \
        .appName("Simulate Streaming") \
        .master("local[8]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Definir el esquema de los archivos CSV
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("y_pred", FloatType(), True),
        StructField("y_test", FloatType(), True)
    ])

    # 3. Leer los archivos de la carpeta como stream
    df = spark.readStream \
        .option("header", True) \
        .schema(schema) \
        .csv("stream_data/file_sink/")  # carpeta, no archivo individual
    
    query = df \
        .writeStream \
        .outputMode('update') \
        .format('console') \
        .start()
    
    query.awaitTermination()

def plot_data_plotly():

    st.title("Predicción vs Real - Simulación de Streaming (Plotly)")

    y_pred_list = []
    y_test_list = []
    x_vals = []

    # Número total de archivos (puedes usar len(y_pred) también si ya lo tienes)
    total_frames = len([f for f in os.listdir('stream_data/file_sink') if f.endswith('.csv')])

    # Inicializar figura
    fig = go.Figure()
    line1 = go.Scatter(x=[], y=[], mode='lines+markers', name='y_pred', line=dict(color='blue'))
    line2 = go.Scatter(x=[], y=[], mode='lines+markers', name='y_test', line=dict(color='green'))
    fig.add_trace(line1)
    fig.add_trace(line2)

    plotly_chart = st.empty()

    for i in range(total_frames):
        try:
            path = f'stream_data/file_sink/ys_{i:05d}.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                x_vals.append(int(df.loc[0, 'id']))
                y_pred_list.append(float(df.loc[0, 'y_pred']))
                y_test_list.append(float(df.loc[0, 'y_test']))
                os.remove(path)

                fig.data[0].x = x_vals
                fig.data[0].y = y_pred_list
                fig.data[1].x = x_vals
                fig.data[1].y = y_test_list

                fig.update_layout(
                    xaxis_title="Muestra",
                    yaxis_title="Valor",
                    title="Predicción vs Real (Streaming)",
                    showlegend=True
                )

                plotly_chart.plotly_chart(fig, use_container_width=True)
                time.sleep(0.5)  # Simula la llegada de datos

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            continue

def main():
    split_csv_file_into_streaming_directory(
        fetch_path=Path('stream_data/ys.csv'),
        streaming_path=Path('stream_data/file_sink')
    )

    threading.Thread(target=initialise_streaming, daemon=True).start()
    plot_data_plotly()

if __name__ == "__main__":
    main()
