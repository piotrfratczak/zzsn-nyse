import torch
import numpy as np
import pandas as pd
from typing import List
from numpy.lib import stride_tricks as st
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from .load_data import load_file


def to_tensor(ndarray):
    return torch.tensor(ndarray).float()


def split_data(stock, seq_len: int, preds_len: int, targets: List[str], val_per100: int = 10, test_per100: int = 10):
    data = st.sliding_window_view(stock, (seq_len + preds_len, stock.shape[1]), axis=(0, 1), subok=True)[:-1, 0]

    val_size = int(np.round(val_per100 / 100 * data.shape[0]))
    test_size = int(np.round(test_per100 / 100 * data.shape[0]))
    train_size = data.shape[0] - (val_size + test_size)
    targets_idx = [stock.columns.get_loc(target) for target in targets if target in stock]

    x_train = data[:train_size, :-preds_len, :]
    y_train = data[:train_size, -preds_len:, targets_idx] \
        .reshape(train_size, preds_len, len(targets))

    x_val = data[train_size:train_size + val_size, :-preds_len, :]
    y_val = data[train_size:train_size + val_size, -preds_len:, targets_idx] \
        .reshape(val_size, preds_len, len(targets))

    x_test = data[train_size + val_size:, :-preds_len, :]
    y_test = data[train_size + val_size:, -preds_len:, targets_idx] \
        .reshape(-1, preds_len, len(targets))

    return x_train, y_train, x_val, y_val, x_test, y_test


def pick_stock(df, symbol):
    df = df[df['symbol'] == symbol]
    df = df.drop('symbol', axis=1)
    return df


def prepare_dataset():
    dataset = load_file('nyse/prices-split-adjusted.csv')
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset.sort_values('date', inplace=True)
    dataset.drop('date', axis=1, inplace=True)
    return dataset


def to_dataloaders(datasets, batch_size):
    x_train, y_train, x_val, y_val, x_test, y_test = datasets

    train_set = TensorDataset(to_tensor(x_train), to_tensor(y_train))
    val_set = TensorDataset(to_tensor(x_val), to_tensor(y_val))
    test_set = TensorDataset(to_tensor(x_test), to_tensor(y_test))

    train_loader = DataLoader(train_set, batch_size, shuffle=True)  # TODO: numworkers?
    val_loader = DataLoader(val_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def concatenate_stocks(dataset, columns, seq_len, preds_len, targets):
    symbols = list(set(dataset.symbol))  # TODO: pick stock?
    x_train = x_val = x_test = np.empty((1, seq_len, len(columns)))
    y_train = y_val = y_test = np.empty((1, 1, 1))
    for symbol in symbols:
        stock_df = pick_stock(dataset, symbol)
        x_train_s, y_train_s, x_val_s, y_val_s, x_test_s, y_test_s = \
            split_data(stock_df[columns], seq_len, preds_len, targets)
        x_train = np.concatenate((x_train, x_train_s))
        y_train = np.concatenate((y_train, y_train_s))
        x_val = np.concatenate((x_val, x_val_s))
        y_val = np.concatenate((y_val, y_val_s))
        x_test = np.concatenate((x_test, x_test_s))
        y_test = np.concatenate((y_test, y_test_s))
    return x_train, y_train, x_val, y_val, x_test, y_test


class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, columns: List[str], targets: List[str], seq_len: int, preds_len: int, batch_size: int):
        columns = list(set(columns).union(set(targets)))
        dataset = prepare_dataset()
        dataset = self.normalize_columns(dataset, columns)

        split_dataset = concatenate_stocks(dataset, columns, seq_len, preds_len, targets)
        train_loader, val_loader, test_loader = to_dataloaders(split_dataset, batch_size)
        return train_loader, val_loader, test_loader

    def normalize_columns(self, df, columns):
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def inverse_normalize(self, df):
        return self.scaler.inverse_transform(df)
