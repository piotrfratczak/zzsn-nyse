from typing import List

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view

from .load_data import load_file


def tensorize(ndarray):
    return torch.tensor(ndarray).float()


def split_data(stock, seq_len: int, preds_len: int, targets: List[str], val_per100: int = 10, test_per100: int = 10):
    data = sliding_window_view(stock, (seq_len + preds_len, stock.shape[1]), axis=(0, 1), subok=True)[:-1, 0]

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


class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, columns: List[str], targets: List[str], seq_len: int, preds_len: int, batch_size: int):
        columns = list(set(columns).union(set(targets)))

        dataset = load_file('nyse/prices-split-adjusted.csv')
        symbols = list(set(dataset.symbol))

        dataset = self.prepare_stock(dataset, symbols[0])
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(dataset[columns], seq_len, preds_len, targets)
        for symbol in symbols[1:]:
            dataset = self.prepare_stock(dataset, symbol)
            xtrain, ytrain, xval, yval, xtest, ytest = split_data(dataset[columns], seq_len, preds_len, targets)
            x_train = np.concatenate((x_train, xtrain))
            y_train = np.concatenate((y_train, ytrain))
            x_val = np.concatenate((x_val, xval))
            y_val = np.concatenate((y_val, yval))
            x_test = np.concatenate((x_test, xtest))
            y_test = np.concatenate((y_test, ytest))

        train_set = TensorDataset(tensorize(x_train), tensorize(y_train))
        val_set = TensorDataset(tensorize(x_val), tensorize(y_val))
        test_set = TensorDataset(tensorize(x_test), tensorize(y_test))

        train_loader = DataLoader(train_set, batch_size, shuffle=True)  # TODO: numworkers?
        val_loader = DataLoader(val_set, batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    def normalize_data(self, df):
        df[:] = self.scaler.fit_transform(df)
        return df

    def inverse_normalize(self, df):
        return self.scaler.inverse_transform(df)

    def prepare_stock(self, df, symbol):
        # TODO: multiple stocks
        df = df[df['symbol'] == symbol]
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.drop(['symbol', 'date'], axis=1, inplace=True)
        df = self.normalize_data(df)
        return df
