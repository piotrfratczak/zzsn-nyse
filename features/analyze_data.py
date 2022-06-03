import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utils.load_data import *
from features.make_dataset import calc_log_returns


def analyze():
    df = load_file('nyse/prices-split-adjusted.csv')

    print(df.shape)
    print(df.head())
    print(df.tail())
    print('--------')
    print(df.describe())
    print('--------')
    print(df.info())
    print('--------')
    print(f'Missing values: {df.isnull().sum().sum()}')
    print('First 10 columns:')
    symbols = list(set(df.symbol))
    print(*symbols[:10], sep=', ')

    log_returns = df.copy()
    log_returns['date'] = pd.to_datetime(log_returns['date'])
    log_returns.sort_values(['symbol', 'date'], inplace=True)
    price_columns = ['open', 'close', 'high', 'low']
    numeric_columns = ['open', 'close', 'high', 'low', 'volume']
    log_returns[price_columns] = calc_log_returns(log_returns[price_columns])
    log_returns = log_returns.iloc[1:]
    nc = log_returns[numeric_columns]
    log_returns[numeric_columns] = (nc - nc.mean()) / nc.std()
    plt.figure(figsize=(10, 10))
    plt.title('Macierz korelacji zmiennych uczących', fontsize=20)
    sns.heatmap(log_returns.corr(),
                annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, linewidth=3, linecolor='black', square=True)
    plt.show()

    plot_symbol = 'GOOG'
    plot_df = df[df['symbol'] == plot_symbol]
    plt.figure(figsize=(22, 18))
    x = plot_df.date
    plt.subplot(2, 1, 1)
    plt.plot(x, plot_df.open.values, color="red", label="Opening Price")
    plt.plot(x, plot_df.close.values, color="blue", label="Closing Price")
    plt.title(f"Stock Prices {plot_symbol} 2010-2016", fontsize=18)
    plt.ylabel("Stock Prices in USD", fontsize=18)
    plt.legend(loc="best")
    plt.grid(which="major", axis="both")
    plt.xticks(x[::50],  rotation='vertical')

    plt.subplot(2, 1, 2)
    plt.plot(x, plot_df.volume.values, color="green", label="Stock Volume Available")
    plt.title(f"Stock Volume of {plot_symbol} 2010-2016", fontsize=18)
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Volume", fontsize=18)
    plt.legend(loc="best")
    plt.grid(which="major", axis="both")
    plt.xticks(x[::50],  rotation='vertical')
    plt.show()

    plt.figure(figsize=(22, 12))
    plot_returns = log_returns[log_returns['symbol'] == plot_symbol]
    plt.plot(plot_returns.date, plot_returns.close, color="violet", label="Logarithmic Returns")
    plt.title(f"Stock Logarighmic Returns of {plot_symbol} 2010-2016", fontsize=18)
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Standardized Log Returns", fontsize=18)
    plt.legend(loc="best")
    plt.grid(which="major", axis="both")
    plt.xticks(x[::50],  rotation='vertical')
    plt.show()

    plot_acf(plot_returns['close'])
    plt.show()

    plot_pacf(plot_returns['close'], method='ywm')
    plt.grid()
    plt.minorticks_on()
    plt.grid(linestyle='--', which='minor', axis='x')
    plt.show()


if __name__ == '__main__':
    analyze()
