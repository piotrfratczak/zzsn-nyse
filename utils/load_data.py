import os
import pandas as pd


DATAFOLDER = '../data/'


def load_file(filename: str) -> pd.DataFrame:
    filepath = os.path.join(os.path.dirname(__file__), DATAFOLDER, filename)
    file = pd.read_csv(filepath)
    return file


def display_files():
    for dirname, _, filenames in os.walk(DATAFOLDER):
        filenames = filter(lambda fname: fname.endswith('.csv'), filenames)
        for filename in filenames:
            print(os.path.join(dirname, filename))
