import polars as pl

from pathlib import Path
import os

def raw_data_from_polars_dataframe(path: Path) -> pl.DataFrame:
    """
    Reads a CSV file into a Polars DataFrame, removes rows with null values, and returns the cleaned DataFrame.

    :param path: Path to the CSV file to be read.
    :type path: Path
    :returns: A Polars DataFrame containing the data from the CSV file with all rows containing null values removed.
    :rtype: pl.DataFrame
    """
    df = pl.read_csv(source=path, has_header=True, separator=',')
    df = df.drop_nulls()
    return df

def transform_polars_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transforms a Polars DataFrame by applying several conversions and feature extractions.
    Performs the following operations:
    - Converts the 'fecha' column from string to date (`%Y-%m-%d`).
    - Converts the 'hora' column from string to time (`%H:%M:%S`).
    - Replaces commas with dots in the 'prec' column and casts it to float.
    - Extracts year, month, day, and day of the week from the 'fecha' column.
    - Adds a boolean column 'is_weekend' indicating if the day is a weekend.
    - Adds a boolean column 'is_holiday' indicating if the day is a holiday (according to certain rules).
    - Casts several columns to float32 type.
    - Drops the 'fecha' and 'error' columns.
    :param df: Polars DataFrame with the original columns.
    :type df: pl.DataFrame
    :return: Transformed DataFrame with new columns and data types.
    :rtype: pl.DataFrame
    """
    
    df = df.with_columns([
        pl.col('fecha').str.to_date('%Y-%m-%d').alias('fecha'),
        pl.col('hora').str.to_time('%H:%M:%S').alias('hora'),
        pl.col('prec').str.replace(',', '.').cast(pl.Float32).alias('prec')
    ])
    
    df = df.with_columns([
        pl.col('fecha').dt.year().alias('year'),
        pl.col('fecha').dt.day().alias('day'),
        pl.col('fecha').dt.weekday().alias('day_of_the_week')
    ])
        
    
    df = df.with_columns([
        pl.col('day_of_the_week').is_in([5,6]).cast(pl.Int8).alias('is_weekend'),

        (
            (pl.col('day').is_in([6,8,25])) | 
            ((pl.col("year") == 2024) & (pl.col("day") == 9))
        ).cast(pl.Float32).alias('is_holiday'),
         
  ])

    # cols_to_convert = ['id', 'intensidad', 'ocupacion', 'carga', 'periodo_integracion', 'year', 'day', 'day_of_the_week', 'is_weekend']
    # df = df.with_columns([
    #     pl.col(c).cast(pl.Float32) for c in cols_to_convert
    # ])

    df = df.drop(['id', 'fecha', 'error', 'periodo_integracion']).remove(pl.col('tipo_elem') == 'C30').sort('day', 'hora', 'distrito', descending=[False, False, False])

    return df

def dataframe_to_parquet(df: pl.DataFrame, path: Path):
    """
    Converts a Polars DataFrame to a Parquet file at the specified path if it does not already exist,
    then reads the Parquet file and returns it as a DataFrame.

    :param df: The Polars DataFrame to be written to Parquet.
    :type df: pl.DataFrame
    :param path: The file system path where the Parquet file will be saved.
    :type path: Path
    :returns: The DataFrame read from the Parquet file at the specified path.
    :rtype: pl.DataFrame
    """

    if not os.path.exists(path):
        df.write_parquet(file=path)

    df_ = pl.read_parquet(source=path)
    return df_
