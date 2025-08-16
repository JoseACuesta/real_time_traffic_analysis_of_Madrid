import pandas as pd
import polars as pl
import requests
import urllib
import json
import time
import glob
import os
from pathlib import Path
from dotenv import load_dotenv

from data_preprocessing.aemet_client import get_precipitation_data_from_aemet


def generate_traffic_data_file(path: Path) -> pl.DataFrame:
    """
    Generates or loads a traffic data file as a polars DataFrame.
    If the specified file does not exist at the given path, this function reads all CSV files matching
    the pattern "data/traffic/*.csv", and saves the resulting DataFrame to the specified path.
    If the file already exists, it loads the DataFrame from disk.
    :param path: Path to the output CSV file where the processed DataFrame will be saved or loaded from.
    :type path: Path
    :return: DataFrame containing the processed traffic data.
    :rtype: pl.DataFrame
    :raises OSError: If there is an error reading or writing files.
    """

    literal_path = "data/traffic/*.csv"
    
    data = []

    columns = [
        "id",
        "fecha",
        "tipo_elem",
        "intensidad",
        "ocupacion",
        "carga",
        "vmed",
        "error",
        "periodo_integracion",
    ]
    
    if not os.path.exists(path):
        print(f"El fichero no existe en {path}")
        try:
            for file in glob.glob(literal_path):
                df = pl.read_csv(source=file, separator=';', has_header=True, columns=columns, null_values="NaN")
                df = df.with_columns([
                pl.col('fecha').str.split_exact(by=' ', n=1)
                .struct.rename_fields(['fecha', 'hora'])
                .alias('split')
                ]).drop('fecha').unnest('split')
                data.append(df)

            df = pl.concat(data).unique()

            new_columns = columns + ["hora"]
            df.select(new_columns).write_parquet(path)
            
            return df

        except OSError as error:
            print(f"Error: {error}")

    df = pl.read_parquet(path)
    return df


def get_data_from_pmed_ubicacion_file(path: Path) -> pl.DataFrame:
    """
    Reads a CSV file containing measurement point locations and returns a DataFrame with selected columns.
    Parameters
    :param path The file path to the CSV file containing measurement point location data.
    :type path: Path
    :return: A DataFrame with the columns 'distrito' and 'id', with any rows containing missing values removed.
    :rtype: pl.Dataframe
    """

    columns = ['distrito', 'id']
    measure_points_data = pl.read_csv(
        source=path, separator=';', has_header=True, encoding='utf8-lossy'
    )
    df = measure_points_data.select(columns).filter(
        pl.col('distrito') == 3.0
        ).drop_nulls()
    return df


def merge_traffic_and_pmed_ubicacion_data(
    traffic_data: pl.DataFrame, pmed_data: pl.DataFrame
) -> pl.DataFrame:
    """
    Merges traffic data with PMED location data based on the 'id' column.

    :param traffic_data: DataFrame containing traffic data with an 'id' column.
    :type traffic_data: pd.DataFrame
    :param pmed_data: DataFrame containing PMED location data with an 'id' column.
    :type pmed_data: pd.DataFrame
    :return: Merged DataFrame containing data from both input DataFrames where 'id' matches.
    :rtype: pd.DataFrame
    """
    df = pmed_data.join(other=traffic_data, on='id', how='left')
    return df


def get_final_data(df: pd.DataFrame, aemet_data: pd.DataFrame, path: Path) -> pl.DataFrame:
    """
    Merges the input DataFrame with AEMET weather data, sorts the result, and saves it to a CSV file.

    :param df: The main DataFrame containing traffic data, with a 'fecha' column.
    :type df: pl.DataFrame
    :param aemet_data: The DataFrame containing AEMET weather data, also with a 'fecha' column.
    :type aemet_data: pl.DataFrame
    :param path: Path to the output CSV file where the processed DataFrame will be saved or loaded from.
    :return: The merged and sorted DataFrame containing both traffic and weather data.
    :rtype: pd.DataFrame
    """

    if not os.path.exists(path):
        df = df.join(other=aemet_data, on='fecha', how='left')
        df = df.sort(by=['id', 'fecha', 'hora'], descending=False)
        df = df.remove(pl.col('id') == 479309)
        df.write_parquet(file=path)
        return df
    
    return pl.read_parquet(source=path)
