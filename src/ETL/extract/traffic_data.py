import io
import zipfile
from urllib.parse import urljoin

import requests
import structlog

logger = structlog.get_logger(__name__)


def extract_historical_traffic_data_from_open_data(file_id: int) -> zipfile.ZipFile:

    for i in range(1, file_id + 1):
        suffix = "-zip" if i < 153 else ""
        BASE_URL = f"https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/resource/208627-{i}-transporte-ptomedida-historico{suffix}/"
        DOWNLOAD_PATH_URL = (
            f"download/208627-{i}-transporte-ptomedida-historico{suffix}.zip"
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
