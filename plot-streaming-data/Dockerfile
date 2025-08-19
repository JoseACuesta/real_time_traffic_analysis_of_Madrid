from python:3.12-slim

WORKDIR /code/plot-streaming-data/

COPY pyproject.toml .
COPY src/plot_streaming_data ./src/plot_streaming_data
COPY src/plot_streaming_data/data ./data

RUN python -m pip install .

ENTRYPOINT [ "python", "-m", "streamlit", "run", "src/plot_streaming_data/homepage.py" ]