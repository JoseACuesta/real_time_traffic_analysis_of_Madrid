# Real Time Traffic Analysis of Madrid

This project is presented as a solution to the Final Master's Project challenge for the Master's degree in Continuing Education in Data Engineering, Big Data and AI.

The main objective of this solution is to analyse Madrid's traffic in December 2021, 2022 and 2023, in order to train models to predict December 2024's traffic and detect anomalies.

A data point is considered an anomaly if the absolute value of the difference between the actual and predicted values is greater than 20% of the actual value.

The results will finally be displayed in a multi-page Streamlit application, where the predictions obtained by the trained Random Forest Regressor and XGBoost Regressor models can be viewed.


This README file outlines the setup required to run the project, provides instructions for doing so, and explains the function of each module.

# Quick Start

The files required to run this project can be found in the following [GitHub repository](https://github.com/JoseACuesta/TFM_data.git)

The README.md file specifies the directory in which each file should be located.

Once you have done so, run the following Docker command:

Get up and running in minutes with Docker:

```bash
# Clone the repository
git clone <repo-url>
cd python-challenge-ml-uv

# Start all services
docker-compose up --build
```

Access the aplication:

- **MinIO Console**: http://localhost:9001
- **Frontend**: http://localhost:8501

## Installation

### Prerequisites

- Python 3.12.7+
- Docker & Docker Compose
- Git

## Modules

### data-preprocessing

This module retrieves traffic data for the Chamberí district in Madrid from December 2021 to December 2024 from the [Madrid council's open data portal](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=02f2c23866b93410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD), as well as rainfall data for those months from [AEMET's open data portal](https://www.aemet.es/es/datos_abiertos).

Unnecessary data is removed and filtered to obtain data for December in the Chamberí district, after which it is combined into a single dataset that will later be used to train machine learning models.

#### Set up

```bash
cd data-preprocessing/src/data-preprocessing

# With uv (uv will take care of creating the virtual environment and installing the dependencies on its own.)
uv run main.py
```

### ml

In this module, feature engineering is carried out to obtain optimal data for correctly training the models.

Once the models have been trained, validated and tested, the datasets, metrics and best model are stored in MinIO.

#### Set up

```bash
cd ml/src/ml

# With uv (uv will take care of creating the virtual environment and installing the dependencies on its own.)
uv run main.py
```

### plot-streaming-data

This module uses Streamlit to display real traffic data from Madrid in December 2024, alongside the test data obtained by each model, to enable visual detection of anomalies.

In addition to Streamlit, other libraries are used to achieve this, such as Plotly, which integrates perfectly with Streamlit, and Spark Structured Streaming, which is used to read data in real time.

#### Set up

```bash
cd ml/src/ml

# With uv (uv will take care of creating the virtual environment and installing the dependencies on its own.)
uv run streamlit run homepage.py
```