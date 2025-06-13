from pathlib import Path
import pandas as pd

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=path, index_col=0)
    return df

def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    pass

if __name__ == "__main__":
    df = load_data(path=Path("../../../data-preprocessing/src/data_preprocessing/data/final_data.csv"))
    print(df.head())