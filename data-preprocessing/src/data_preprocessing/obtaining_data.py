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
                df = pl.read_parquet(source=file, separator=';', has_header=True, new_columns=columns, null_values="NaN")
                df = df.with_columns([
                pl.col('fecha').str.split_exact(by=' ', n=1)
                .struct.rename_fields(['fecha', 'hora'])
                .alias('split')
                ]).drop('fecha').unnest('split')
                data.append(df)

            df = pl.concat(data).unique()

            new_columns = columns + ["hora"]
            df.select(new_columns).write_csv(path, separator=';')
            
            return df

        except OSError as error:
            print(f"Error: {error}")

    df = pl.read_csv(path, separator=';')
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
        pl.col('distrito') == 1.0
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


def get_precipitation_data_from_aemet(path: Path) -> pl.DataFrame:
    """This function checks if a local CSV file containing historical precipitation data exists.
    If not, it fetches the data from the AEMET API in several date ranges, concatenates the results, removes duplicates,
    and saves the data to a CSV file for future use.
    If the file already exists, it loads the data directly from the CSV.

    :param path: The file path to the CSV file containing historical precipitation data.
    :type path: Path
    :return: DataFrame containing two columns: 'fecha' (date) and 'prec' (precipitation), with daily precipitation data for the specified station.
    :rtype: pl.DataFrame

    :raises: Prints error messages if there are issues with the API request or data retrieval.
    """

    load_dotenv()

    main_dataframe = pl.DataFrame()

    dates = [
        "2021-01-01T00:00:00UTC",
        "2021-07-01T00:00:00UTC",
        "2022-01-01T00:00:00UTC",
        "2022-07-01T00:00:00UTC",
        "2023-01-01T00:00:00UTC",
        "2023-07-01T00:00:00UTC",
        "2024-01-01T00:00:00UTC",
        "2024-07-01T00:00:00UTC",
        "2025-01-01T00:00:00UTC"
    ]

    API_KEY = os.getenv("AEMET_API_KEY")
    BASE_URL = "https://opendata.aemet.es/opendata"
    IDEMA = "3195"

    if not path.exists():
        for i in range(len(dates) - 1):
            fechaIniStr = dates[i]
            fechaFinStr = dates[i+1]

            ENDPOINT = f"/api/valores/climatologicos/diarios/datos/fechaini/{fechaIniStr}/fechafin/{fechaFinStr}/estacion/{IDEMA}"
            URL = BASE_URL + ENDPOINT
            headers = {"Accept": "application/json", "api_key": API_KEY}

            try:
                response = requests.get(URL, headers=headers, timeout=30)
                data = response.json()

                if data.get('datos') is not None:
                    weather_data = data['datos']
                    file = urllib.request.urlopen(weather_data)
                    file_content = file.read()
                    weather_data_json = json.loads(file_content)
                    secondary_dataframe = pl.DataFrame(data=weather_data_json)
                    main_dataframe = pl.concat(
                        [main_dataframe, secondary_dataframe], how='diagonal'
                    )
                    secondary_dataframe = pl.DataFrame()
                    time.sleep(5)
                else:
                    print(
                        f"No se encontró la clave 'datos' en la respuesta para fechas {fechaIniStr} a {fechaFinStr}"
                    )
                    print(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                print(
                    f"Error durante la solicitud para {fechaIniStr} a {fechaFinStr}: {e}"
                )

        main_dataframe = main_dataframe.unique()
        main_dataframe.write_csv(file=path)

        df = main_dataframe.select([
            pl.col('fecha'),
            pl.col('prec').fill_null('0.0')
        ])
        return df
    
    df = pl.read_csv(path).select([
        pl.col('fecha'),
        pl.col('prec').fill_null('0.0')
    ])
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
        df = df.sort(by=['id', 'fecha'], descending=False)
        df = df.remove(pl.col('id') == 479309)
        df.write_csv(file=path)
        return df
    
    return pl.read_csv(source=path)


def generate_final_dataframe():
    initial_traffic_data = generate_traffic_data_file(
        path=Path("data/traffic/historic_traffic_data_december.csv")
    )
    pmed_ubicacion_data = get_data_from_pmed_ubicacion_file(
        path=Path("data/pmed_ubicacion_04_2025.csv")
    )
    data = merge_traffic_and_pmed_ubicacion_data(
        traffic_data=initial_traffic_data, pmed_data=pmed_ubicacion_data
    )
    precipitation_data = get_precipitation_data_from_aemet(path=Path("data/historic_aemet_data.csv"))
    df = get_final_data(df=data, aemet_data=precipitation_data, path=Path('data/provisional_final_data.csv'))
    print(df.shape)
    