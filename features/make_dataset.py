import torch
import numpy as np
import pandas as pd
from typing import List
from numpy.lib import stride_tricks as st
from torch.utils.data import TensorDataset, DataLoader

from utils.load_data import load_file
from utils.setup import get_features


def calc_log_returns(prices):
    return (np.log(prices) - np.log(prices.shift(1)))[1:]


def to_tensor(ndarray):
    return torch.tensor(ndarray).float()


def slide_window(df, seq_len, proj_len):
    return st.sliding_window_view(df, (seq_len + proj_len, df.shape[1]), axis=(0, 1), subok=True)[:-1, 0]


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


def to_loaders(datasets: tuple, batch_size: int):
    x_train, y_train, x_val, y_val, x_test, y_test = datasets

    train_set = TensorDataset(to_tensor(x_train), to_tensor(y_train))
    val_set = TensorDataset(to_tensor(x_val), to_tensor(y_val))
    test_set = TensorDataset(to_tensor(x_test), to_tensor(y_test))

    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader


class DatasetMaker:
    def __init__(self):
        self.scalers = dict()

    def make_dataset(self, args):
        features, targets = get_features(args)
        features = list(set(features).union(set(targets)))
        dataset = prepare_dataset()
        split_dataset = self.concatenate_stocks(dataset, features, targets, args.seq_len, args.proj_len)
        train_loader, val_loader, test_loader = to_loaders(split_dataset, args.batch_size)
        return train_loader, val_loader, test_loader

    def standardize(self, df: pd.DataFrame, symbol: str):
        mean, std = df.mean(), df.std()
        self.scalers.update({symbol: (mean, std)})
        return (df - mean) / std

    def inverse_standardize(self, df: pd.DataFrame, symbol: str):
        mean, std = self.scalers[symbol]
        return df * std + mean

    def concatenate_stocks(self, dataset: pd.DataFrame, features: List[str], targets: List[str], seq_len: int, proj_len: int):
        symbols = list(set(dataset.symbol))
        stock_df = pick_stock(dataset, symbols[0])
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_data(stock_df[features], symbols[0], seq_len, proj_len, targets)

        for symbol in symbols[1:]:
            stock_df = pick_stock(dataset, symbol)
            x_train_s, y_train_s, x_val_s, y_val_s, x_test_s, y_test_s = \
                self.split_data(stock_df[features], symbol, seq_len, proj_len, targets)

            x_train = np.concatenate((x_train, x_train_s))
            y_train = np.concatenate((y_train, y_train_s))
            x_val = np.concatenate((x_val, x_val_s))
            y_val = np.concatenate((y_val, y_val_s))
            x_test = np.concatenate((x_test, x_test_s))
            y_test = np.concatenate((y_test, y_test_s))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def split_data(self, stock: pd.DataFrame, symbol: str, seq_len: int, proj_len: int, targets: List[str], val_split: int = 10, test_split: int = 10):
        val_size = int(np.round(val_split / 100 * stock.shape[0]))
        test_size = int(np.round(test_split / 100 * stock.shape[0]))
        train_size = stock.shape[0] - (val_size + test_size)

        log_returns = calc_log_returns(stock)
        standardized_log_returns = self.standardize(log_returns, symbol)

        train_series = standardized_log_returns[:train_size]
        val_series = standardized_log_returns[train_size: train_size + val_size]
        test_series = standardized_log_returns[train_size + val_size:]

        train_windows = slide_window(train_series, seq_len, proj_len)
        val_windows = slide_window(val_series, seq_len, proj_len)
        test_windows = slide_window(test_series, seq_len, proj_len)

        targets_idx = [stock.columns.get_loc(target) for target in targets if target in stock]

        x_train = train_windows[:, :-proj_len, :]
        y_train = train_windows[:, -proj_len:, targets_idx]
        x_val = val_windows[:, :-proj_len, :]
        y_val = val_windows[:, -proj_len:, targets_idx]
        x_test = test_windows[:, :-proj_len, :]
        y_test = test_windows[:, -proj_len:, targets_idx]

        return x_train, y_train, x_val, y_val, x_test, y_test
