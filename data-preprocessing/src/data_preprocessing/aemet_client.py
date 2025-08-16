import polars as pl
import requests
import os
from pathlib import Path
from dotenv import load_dotenv


def get_precipitation_data_from_aemet(path: Path) -> pl.DataFrame:

    load_dotenv()

    list_aemet_data = []

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

    API_KEY = os.environ.get('AEMET_API_KEY')
    BASE_URL = "https://opendata.aemet.es/opendata"
    IDEMA = "3195"

    if not os.path.exists(path=path):

        for i in range(len(dates)-1):
            fechaIniStr = dates[i]
            fechaFinStr = dates[i+1]

            ENDPOINT = f'/api/valores/climatologicos/diarios/datos/fechaini/{fechaIniStr}/fechafin/{fechaFinStr}/estacion/{IDEMA}'
            URL = BASE_URL + ENDPOINT

            headers = {
                'Accept': 'Application/json',
                'api_key': API_KEY
            }

            try:
                response = requests.get(url=URL, headers=headers, timeout=30)
                response.raise_for_status()
                response_json = response.json()
                
                data_url = response_json['datos']

                try:
                    response_data = requests.get(url=data_url, timeout=30)
                    response_data.raise_for_status()
                    weather_data = response_data.json()
                    weather_dataframe = pl.DataFrame(weather_data).select(
                        pl.col('fecha'),
                        pl.col('prec').fill_null('0.0')
                    )
                    list_aemet_data.append(weather_dataframe)
                    weather_dataframe = pl.DataFrame()

                except requests.exceptions.RequestException as e:
                    print(f'Error de conexión para {fechaFinStr} a {fechaFinStr}: {e}')
            
            except Exception as e:
                print(
                    f"Error durante la solicitud para {fechaIniStr} a {fechaFinStr}: {e}"
                )
        
        aemet_dataframe = pl.concat(items=list_aemet_data)
        aemet_dataframe = aemet_dataframe.unique()
        aemet_dataframe.write_parquet(file=path)

        return aemet_dataframe

    else:
        return pl.read_parquet(source=path)
