import os
import pathlib
import pandas as pd


data_dir = os.path.join(pathlib.Path(__file__).parent.parent, 'data')


def load_file(filename: str) -> pd.DataFrame:
    filepath = os.path.join(os.path.dirname(__file__), data_dir, filename)
    file = pd.read_csv(filepath)
    return file


def display_files():
    for dirname, _, filenames in os.walk(data_dir):
        filenames = filter(lambda fname: fname.endswith('.csv'), filenames)
        for filename in filenames:
            print(os.path.join(dirname, filename))
