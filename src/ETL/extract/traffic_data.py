import io
import zipfile
from urllib.parse import urljoin

import requests
import structlog

logger = structlog.get_logger(__name__)


def extract_historical_traffic_data_from_open_data(file_id: int) -> zipfile.ZipFile:

    for i in range(1, file_id + 1):
        if i < 153:
            BASE_URL = f"https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/resource/208627-{file_id}-transporte-ptomedida-historico-zip/"
            DOWNLOAD_PATH_URL = (
                f"download/208627-{file_id}-transporte-ptomedida-historico-zip.zip"
            )
            FULL_URL = urljoin(BASE_URL, DOWNLOAD_PATH_URL)
            logger.info(f"Full URL: {FULL_URL}")
        else:
            BASE_URL = f"https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/resource/208627-{file_id}-transporte-ptomedida-historico/"
            DOWNLOAD_PATH_URL = (
                f"download/208627-{file_id}-transporte-ptomedida-historico.zip"
            )
            FULL_URL = urljoin(BASE_URL, DOWNLOAD_PATH_URL)
            logger.info(f"Full URL: {FULL_URL}")

        try:
            response = requests.get(url=FULL_URL)
            response.raise_for_status()
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    month_year_file = zip_file.namelist()[0].split(".")[0]
                    logger.info(f"Zip file obtained for month-year: {month_year_file}")
                    return zip_file
        except zipfile.BadZipfile:
            logger.exception(
                f"This is not a valid ZIP file for the required file_id {file_id}: {zip_file}"
            )
            raise
        except FileNotFoundError:
            logger.exception(f"{zip_file} not found for file_id {file_id}")


if __name__ == "__main__":
    extract_historical_traffic_data_from_open_data(file_id=1)
