from datetime import datetime
from dateutil.relativedelta import relativedelta

import os
import structlog
import polars as pl
import requests
from pathlib import Path

from src.common.settings import settings

logger = structlog.get_logger(__name__)

BASE_URL = "https://opendata.aemet.es/opendata"


def get_precipitation_data_from_aemet(start_date: datetime, end_date: datetime) -> list:

    list_aemet_data = []

    API_KEY = str(settings.api_key)
    IDEMA = settings.aemet_station_idema
    headers = {"Accept": "Application/json", "api_key": API_KEY}


    while start_date < end_date:
        if start_date.year == 2020:
            logger.info("Skipping 2020 due to COVID")
            start_date = datetime(2021, 1, 1)
            continue

        chunk_end_date = min(
            start_date + relativedelta(months=6) - relativedelta(days=1),
            end_date
        )

        start_date_parse = start_date.strftime("%Y-%m-%dT%H:%M:%SUTC")
        end_date_parse = chunk_end_date.strftime("%Y-%m-%dT%H:%M:%SUTC")

        ENDPOINT: str = f"/api/valores/climatologicos/diarios/datos/fechaini/{start_date_parse}/fechafin/{end_date_parse}/estacion/{IDEMA}"
        URL: str = BASE_URL + ENDPOINT
        
        try:
            response = requests.get(url=URL, headers=headers, timeout=30)
            logger.info("Fetching data from AEMET API", start_date=start_date, end_date=chunk_end_date)
            response.raise_for_status()
            response_json = response.json()

            data_url = response_json["datos"]

            try:
                response_data = requests.get(url=data_url, timeout=30)
                response_data.raise_for_status()
                weather_data = response_data.json()
                list_aemet_data.extend(weather_data)
                logger.info("Data added succesfully ", data=len(weather_data))
            
            except requests.exceptions.RequestException as e:
                logger.exception(
                    "Connection error",
                    start_date=start_date_parse,
                    end_date=chunk_end_date,
                    error=str(e)
                )

        except Exception as e:
            logger.exception(
                "Error while fetching data",
                start_date=start_date_parse,
                end_date=chunk_end_date,
                error=str(e)
            )
        
        start_date = chunk_end_date + relativedelta(days=1)
    
    return list_aemet_data


