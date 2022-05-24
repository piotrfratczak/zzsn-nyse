import torch
import numpy as np
import pandas as pd
from typing import List
from numpy.lib import stride_tricks as st
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from .load_data import load_file


def calc_log_returns(prices):
    return (np.log(prices) - np.log(prices.shift(1)))[1:]


def to_tensor(ndarray):
    return torch.tensor(ndarray).float()


def slide_window(df, seq_len, pred_len):
    return st.sliding_window_view(df, (seq_len + pred_len, df.shape[1]), axis=(0, 1), subok=True)[:-1, 0]


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


def to_loaders(datasets, batch_size):
    x_train, y_train, x_val, y_val, x_test, y_test = datasets

    train_set = TensorDataset(to_tensor(x_train), to_tensor(y_train))
    val_set = TensorDataset(to_tensor(x_val), to_tensor(y_val))
    test_set = TensorDataset(to_tensor(x_test), to_tensor(y_test))

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


class Preprocessor:
    def __init__(self):
        self.scaler = None

    def preprocess(self, columns: List[str], targets: List[str], seq_len: int, pred_len: int, batch_size: int):
        columns = list(set(columns).union(set(targets)))
        dataset = prepare_dataset()
        self.scaler = StandardScaler().fit(dataset[columns])

        split_dataset = self.concatenate_stocks(dataset, columns, seq_len, pred_len, targets)
        train_loader, val_loader, test_loader = to_loaders(split_dataset, batch_size)
        return train_loader, val_loader, test_loader

    def inverse_standardize(self, df):
        return self.scaler.inverse_transform(df)

    def concatenate_stocks(self, dataset, columns, seq_len, pred_len, targets):
        symbols = list(set(dataset.symbol))  # TODO: pick stock?
        x_train = x_val = x_test = np.empty((1, seq_len, len(columns)))
        y_train = y_val = y_test = np.empty((1, 1, 1))

        for symbol in symbols:
            stock_df = pick_stock(dataset, symbol)
            x_train_s, y_train_s, x_val_s, y_val_s, x_test_s, y_test_s = \
                self.split_data(stock_df[columns], seq_len, pred_len, targets)

            x_train = np.concatenate((x_train, x_train_s))
            y_train = np.concatenate((y_train, y_train_s))
            x_val = np.concatenate((x_val, x_val_s))
            y_val = np.concatenate((y_val, y_val_s))
            x_test = np.concatenate((x_test, x_test_s))
            y_test = np.concatenate((y_test, y_test_s))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def split_data(self, stock, seq_len: int, pred_len: int, targets: List[str], val_per100: int = 10, test_per100: int = 10):
        val_size = int(np.round(val_per100 / 100 * stock.shape[0]))
        test_size = int(np.round(test_per100 / 100 * stock.shape[0]))
        train_size = stock.shape[0] - (val_size + test_size)

        train_series = calc_log_returns(stock[:train_size])
        val_series = calc_log_returns(stock[train_size: train_size + val_size])
        test_series = calc_log_returns(stock[train_size + val_size:])

        scaled_train_series = self.scaler.transform(train_series)
        scaled_val_series = self.scaler.transform(val_series)
        scaled_test_series = self.scaler.transform(test_series)

        train_windows = slide_window(scaled_train_series, seq_len, pred_len)
        val_windows = slide_window(scaled_val_series, seq_len, pred_len)
        test_windows = slide_window(scaled_test_series, seq_len, pred_len)

        targets_idx = [stock.columns.get_loc(target) for target in targets if target in stock]

        x_train = train_windows[:, :-pred_len, :]
        y_train = train_windows[:, -pred_len:, targets_idx]
        x_val = val_windows[:, :-pred_len, :]
        y_val = val_windows[:, -pred_len:, targets_idx]
        x_test = test_windows[:, :-pred_len, :]
        y_test = test_windows[:, -pred_len:, targets_idx]
        return x_train, y_train, x_val, y_val, x_test, y_test
