from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from load_data import *

df = load_file('nyse/prices-split-adjusted.csv')

print(df.shape)
print(df.head())
print(df.tail())

print(df.describe())

plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, linewidth=3, linecolor='black', square=True)
plt.show()

print(df.info())

print(f'Missing values: {df.isnull().sum().sum()}')

symbols = list(set(df.symbol))
print(symbols[:10])

plot_symbol = 'AAPL'
plot_df = df[df['symbol'] == plot_symbol]
plt.figure(figsize=(20, 12))
x = np.arange(0, plot_df.shape[0], 1)
plt.subplot(2, 1, 1)
plt.plot(x, plot_df.open.values, color="red", label="Opening Price")
plt.plot(x, plot_df.close.values, color="blue", label="Closing Price")
plt.title(f"Stock Prices {plot_symbol} 2010-2016", fontsize=18)
plt.xlabel("Days", fontsize=18)
plt.ylabel("Stock Prices in USD", fontsize=18)
plt.legend(loc="best")
plt.grid(which="major", axis="both")

plt.subplot(2, 1, 2)
plt.plot(x, plot_df.volume.values, color="green", label="Stock Volume Available")
plt.title(f"Stock Volume of {plot_symbol} 2010-2016", fontsize=18)
plt.xlabel("Days", fontsize=18)
plt.ylabel("Volume", fontsize=18)
plt.legend(loc="best")
plt.grid(which="major", axis="both")
plt.show()

# TODO: normalize first
plot_acf(plot_df['close'].values[:100])
plt.show()
plot_pacf(plot_df['close'].values[:100], method='ywm')
plt.show()
