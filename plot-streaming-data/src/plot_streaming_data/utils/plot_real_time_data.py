import pandas as pd

import os
import time

import streamlit as st
import plotly.graph_objects as go

def plot_rfr_data_plotly():
    """
    Visualizes in real time the comparison between predicted and actual traffic values using Plotly and Streamlit.
    This function iterates over CSV files generated in real time, extracts prediction and actual values, and displays them in an interactive plot.
    It detects anomalies when the difference between the predicted and actual value exceeds a threshold (20% of the actual value), highlighting them in the plot and showing a warning in the interface.
    Processed files are deleted after visualization.
    :raises Exception: Displays an error message in Streamlit if any issue occurs while processing the files.
    """

    SLEEP = 1

    x_vals_list = []
    y_pred_list = []
    y_test_list = []
    y_pred_colors = []

    total_frames = len([f for f in os.listdir('data/RandomForestRegressor/file_sink') if f.endswith('.csv')])

    plotly_chart = st.empty()
    warning_container = st.container(border=True, height=400)

    for i in range(total_frames):
        try:
            path = f'data/RandomForestRegressor/file_sink/ys_{i:06d}.csv'
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
                    warning_container.warning(body=f'ANOMALY: The value of the load of traffic was supossed to be: {y_test_value}, but {y_pred_value} has been predicted', icon='⚠️')
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
                title="Predicted vs Actual Traffic Values",
                showlegend=True
                )

                plotly_chart.plotly_chart(fig, use_container_width=False, key=f'dinamyc_plot_{i}')
                os.remove(path)
                time.sleep(SLEEP)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            continue

def plot_xgboost_data_plotly():
    """
    Visualizes in real time the comparison between predicted and actual traffic values using Plotly and Streamlit.
    This function iterates over CSV files generated in real time, extracts prediction and actual values, and displays them in an interactive plot.
    It detects anomalies when the difference between the predicted and actual value exceeds a threshold (20% of the actual value), highlighting them in the plot and showing a warning in the interface.
    Processed files are deleted after visualization.
    :raises Exception: Displays an error message in Streamlit if any issue occurs while processing the files.
    """

    SLEEP = 1

    x_vals_list = []
    y_pred_list = []
    y_test_list = []
    y_pred_colors = []

    total_frames = len([f for f in os.listdir('data/XGBoost/file_sink') if f.endswith('.csv')])

    plotly_chart = st.empty()
    warning_container = st.container(border=True, height=200)

    for i in range(total_frames):
        try:
            path = f'data/XGBoost/file_sink/ys_{i:06d}.csv'
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
                    warning_container.warning(body=f'ANOMALY: The value of the load of traffic was supossed to be: {y_test_value}, but {y_pred_value} has been predicted', icon='⚠️')
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
                title="Predicted vs Actual Traffic Values",
                showlegend=True
                )

                plotly_chart.plotly_chart(fig, use_container_width=True, key=f'dinamyc_plot_{i}')
                os.remove(path)
                time.sleep(SLEEP)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            continue
