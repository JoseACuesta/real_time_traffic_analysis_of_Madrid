from pathlib import Path

from data_preprocessing.obtaining_data import (
    generate_traffic_data_file,
    get_data_from_pmed_ubicacion_file,
    merge_traffic_and_pmed_ubicacion_data,
    get_final_data
)

from data_preprocessing.aemet_client import get_precipitation_data_from_aemet

def main():
    initial_traffic_data = generate_traffic_data_file(
    path=Path("data/traffic/historic_traffic_data_december.parquet")
    )
    pmed_ubicacion_data = get_data_from_pmed_ubicacion_file(
        path=Path("data/pmed_ubicacion_04_2025.csv")
    )
    data = merge_traffic_and_pmed_ubicacion_data(
        traffic_data=initial_traffic_data, pmed_data=pmed_ubicacion_data
    )
    precipitation_data = get_precipitation_data_from_aemet(path=Path("data/historic_aemet_data.parquet"))
    df = get_final_data(df=data, aemet_data=precipitation_data, path=Path('data/provisional_final_data.parquet'))

if __name__ == "__main__":
    main()