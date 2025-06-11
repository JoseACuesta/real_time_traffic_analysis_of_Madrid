import pandas as pd
import requests
import urllib
import json
import time
import glob
import os
from pathlib import Path
import io
from dotenv import load_dotenv

def generate_traffic_data_file(path: Path, chunksize:int = 500_000) -> pd.DataFrame:
    
    """
    Generates or loads a traffic data file as a pandas DataFrame.
    If the specified file does not exist at the given path, this function reads all CSV files matching
    the pattern "data/traffic/*.csv", and saves the resulting DataFrame to the specified path.
    If the file already exists, it loads the DataFrame from disk.
    :param path: Path to the output CSV file where the processed DataFrame will be saved or loaded from.
    :type path: Path
    :param chunksize: Number of rows per chunk to read from each CSV file. Default is 500,000.
    :type chunksize: int, optional
    :return: DataFrame containing the processed traffic data.
    :rtype: pd.DataFrame
    :raises OSError: If there is an error reading or writing files.
    """
    
    # MODIFICAR PARA TENER EN CUENTA HEADERS

    literal_path = "data/traffic/*.csv"

    data = []

    columns = ['id','fecha','tipo_elem','intensidad','ocupacion','carga','vmed','error','periodo_integracion']

    if not os.path.exists(path):
        print(f"El fichero no existe en {path}")
        try:
            for file in glob.glob(literal_path):
                for chunk in pd.read_csv(file, chunksize=chunksize, sep=";", names=columns, header=0):
                    data.append(chunk)
            df = pd.concat(data, ignore_index=True).drop_duplicates(keep='first')
            df[['fecha', 'hora']] = df['fecha'].str.split(' ', expand=True)
            new_columns = columns.append('hora')
            df.to_csv(path, columns=new_columns, sep=";", index=False)
            return df
        except OSError as error:
            print(f"Error: {error}")
    
    df = pd.read_csv(path, sep=";")
    return df

def get_data_from_pmed_ubicacion_file(path: Path) -> pd.DataFrame:
    """
    Reads a CSV file containing measurement point locations and returns a DataFrame with selected columns.
    Parameters
    :param path The file path to the CSV file containing measurement point location data.
    :type path: Path
    :return: A DataFrame with the columns 'distrito' and 'id', with any rows containing missing values removed.
    :rtype: pd.Dataframe
    """
    
    columns = ['distrito', 'id']
    measure_points_data = pd.read_csv(filepath_or_buffer=path, sep=';', usecols=columns, encoding='utf-8')
    if measure_points_data.isna().any().any():
        measure_points_data.dropna(inplace=True)
    return measure_points_data

def merge_traffic_and_pmed_ubicacion_data(traffic_data: pd.DataFrame, pmed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merges traffic data with PMED location data based on the 'id' column.

    :param traffic_data: DataFrame containing traffic data with an 'id' column.
    :type traffic_data: pd.DataFrame
    :param pmed_data: DataFrame containing PMED location data with an 'id' column.
    :type pmed_data: pd.DataFrame
    :return: Merged DataFrame containing data from both input DataFrames where 'id' matches.
    :rtype: pd.DataFrame
    """
    df = traffic_data.merge(pmed_data, left_on='id', right_on='id')
    return df

def get_precipitation_data_from_aemet() -> pd.DataFrame:
    def get_precipitation_data_from_aemet():

        """This function checks if a local CSV file containing historical precipitation data exists.
        If not, it fetches the data from the AEMET API in several date ranges, concatenates the results, removes duplicates, 
        and saves the data to a CSV file for future use. 
        If the file already exists, it loads the data directly from the CSV.

        :return: DataFrame containing two columns: 'fecha' (date) and 'prec' (precipitation), with daily precipitation data for the specified station.
        :rtype: pandas.DataFrame

        :raises: Prints error messages if there are issues with the API request or data retrieval.
        """

    load_dotenv()

    main_dataframe = pd.DataFrame()
   
    file_path = Path('data/historic_aemet_data.csv')
 
    dates = ['2021-01-01T00:00:00UTC', '2021-07-01T00:00:00UTC', '2022-01-01T00:00:00UTC', '2022-07-01T00:00:00UTC', '2023-01-01T00:00:00UTC',
             '2023-07-01T00:00:00UTC', '2024-01-01T00:00:00UTC', '2024-07-01T00:00:00UTC', '2025-01-01T00:00:00UTC', '2025-04-30T00:00:00UTC']

    API_KEY = os.getenv('AEMET_API_KEY')
    base_url = 'https://opendata.aemet.es/opendata'
    idema = '3195' # Estación

    if not file_path.exists():
        for i in range(len(dates)-1):
            fechaIniStr = dates[i]
            fechaFinStr = dates[i+1]

            endpoint = f"/api/valores/climatologicos/diarios/datos/fechaini/{fechaIniStr}/fechafin/{fechaFinStr}/estacion/{idema}"
            url = base_url + endpoint
            headers = {
                "Accept": "application/json",
                "api_key": API_KEY
            }

            try:
                response = requests.get(url, headers=headers, timeout=15)
                data = response.json()

                if data.get('datos') is not None:
                    weather_data = data['datos']
                    file = urllib.request.urlopen(weather_data)
                    file_content = file.read()
                    weather_data_json = json.loads(file_content)
                    secondary_dataframe = pd.DataFrame(weather_data_json)
                    # Eliminamos duplicados porque las fechas de los extremos se cogen dos veces cada una salvo la primera y la última
                    main_dataframe = pd.concat([main_dataframe, secondary_dataframe], ignore_index=True).drop_duplicates()
                    secondary_dataframe = secondary_dataframe.fillna(0)
                    time.sleep(2)
                else:
                    print(f"No se encontró la clave 'datos' en la respuesta para fechas {fechaIniStr} a {fechaFinStr}")
                    print(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Error durante la solicitud para {fechaIniStr} a {fechaFinStr}: {e}")

        main_dataframe.to_csv('data/historic_aemet_data.csv', index=False)
        return main_dataframe[['fecha', 'prec']]
    
    else:
        df = pd.read_csv(file_path)[['fecha', 'prec']]
        return df
    
def get_final_data(df: pd.DataFrame, aemet_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the input DataFrame with AEMET weather data, sorts the result, and saves it to a CSV file.

    :param df: The main DataFrame containing traffic data, with a 'fecha' column.
    :type df: pd.DataFrame
    :param aemet_data: The DataFrame containing AEMET weather data, also with a 'fecha' column.
    :type aemet_data: pd.DataFrame
    :return: The merged and sorted DataFrame containing both traffic and weather data.
    :rtype: pd.DataFrame
    """
    
    df = df.merge(aemet_data, left_on='fecha', right_on='fecha')
    df.sort_values(by=['id', 'fecha'], ascending=True, inplace=True)
    df.to_csv('data/final_data.csv')
    return df
    
def main():
    initial_traffic_data = generate_traffic_data_file(path=Path("data/traffic/historic_traffic_data_december.csv"))
    pmed_ubicacion_data = get_data_from_pmed_ubicacion_file(path=Path("data/pmed_ubicacion_04_2025.csv"))
    data = merge_traffic_and_pmed_ubicacion_data(traffic_data=initial_traffic_data, pmed_data=pmed_ubicacion_data)
    precipitation_data = get_precipitation_data_from_aemet()
    df = get_final_data(data, precipitation_data)
    print(df.head())

if __name__ == "__main__":
    main()