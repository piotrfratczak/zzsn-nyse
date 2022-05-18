import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from .load_data import load_file


def preprocess(columns, seq_len, preds_len, batch_size):
    dataset = prepare_stock()
    prices = dataset[columns]

    train_data, train_labels, val_data, val_labels, test_data, test_labels = \
        split_data(prices, seq_len, preds_len, targets=['close'])

    x_train = torch.tensor(train_data).float()
    y_train = torch.tensor(train_labels).float()
    x_val = torch.tensor(val_data).float()
    y_val = torch.tensor(val_labels).float()
    x_test = torch.tensor(test_data).float()
    y_test = torch.tensor(test_labels).float()

    # TODO: delete shape printing
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_val.shape = ', x_val.shape)
    print('y_val.shape = ', y_val.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    train_set = TensorDataset(x_train, y_train)
    val_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)  # TODO: numworkers?
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def normalize_data(df):
    scaler = MinMaxScaler()
    columns = df.columns.drop(['date', 'symbol'])
    df[columns] = scaler.fit_transform(df[columns])
    return df


def prepare_stock(symbol='AAPL'):
    df = load_file('nyse/prices.csv')
    df = df[df['symbol'] == symbol]
    df = normalize_data(df)
    df.drop(['symbol'], axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    return df


def split_data(stock, seq_len, preds_len, targets):
    data = []
    for first_day in range(len(stock) - seq_len - preds_len):
        data.append(stock[first_day: first_day + seq_len + preds_len])
    data = np.array(data)

    val_percentage = 10
    test_percentage = 10

    val_size = int(np.round(val_percentage / 100 * data.shape[0]))
    test_size = int(np.round(test_percentage / 100 * data.shape[0]))
    train_size = data.shape[0] - (val_size + test_size)

    targets_idx = [stock.columns.get_loc(target) for target in targets if target in stock]

    x_train = data[:train_size, :-preds_len, :]
    y_train = data[:train_size, -preds_len:, targets_idx].reshape(train_size, preds_len, len(targets))

    x_val = data[train_size:train_size + val_size, :-preds_len, :]
    y_val = data[train_size:train_size + val_size, -preds_len:, targets_idx].reshape(val_size, preds_len, len(targets))

    x_test = data[train_size + val_size:, :-preds_len, :]
    y_test = data[train_size + val_size:, -preds_len:, targets_idx].reshape(-1, preds_len, len(targets))

    return x_train, y_train, x_val, y_val, x_test, y_test

