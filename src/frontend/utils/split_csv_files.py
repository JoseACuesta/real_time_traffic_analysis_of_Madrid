from pathlib import Path
import os


def split_rfr_prediction_data_file_into_streaming_directory(
    file_path: Path, streaming_directory: Path
):
    """
    Splits a CSV file into multiple single-row CSV files in a specified directory.
    :param file_path: Path to the input CSV file to be split.
    :type file_path: Path
    :param streaming_directory: Directory where the split CSV files will be saved.
    :type streaming_directory: Path
    """

    MAX_FILES = 5000
    with open(file=file_path, mode="r") as file_:
        os.makedirs(name=streaming_directory, exist_ok=True)
        header = file_.readline()
        for i, line in enumerate(file_):
            if i <= MAX_FILES:
                path = f"{streaming_directory}/ys_{i:06d}.csv"
                with open(file=path, mode="w") as f:
                    f.write(header)
                    f.write(line)


def split_xgboost_prediction_data_file_into_streaming_directory(
    file_path: Path, streaming_directory: Path
):
    """
    Splits a CSV file into multiple single-row CSV files in a specified directory.
    :param file_path: Path to the input CSV file to be split.
    :type file_path: Path
    :param streaming_directory: Directory where the split CSV files will be saved.
    :type streaming_directory: Path
    """

    MAX_FILES = 5000
    with open(file=file_path, mode="r") as file_:
        os.makedirs(name=streaming_directory, exist_ok=True)
        header = file_.readline()
        for i, line in enumerate(file_):
            if i <= MAX_FILES:
                path = f"{streaming_directory}/ys_{i:06d}.csv"
                with open(file=path, mode="w") as f:
                    f.write(header)
                    f.write(line)
