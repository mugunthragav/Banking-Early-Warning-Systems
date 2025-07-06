# src/visualization/plots.py
import matplotlib.pyplot as plt
import os


def plot_returns(returns, symbol):
    """
    Plot cumulative returns for a given symbol and save to output directory.

    Args:
        returns (pd.Series): Series of returns.
        symbol (str): Symbol name for the plot title and file name.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(returns.cumsum(), label='Cumulative Returns')
    plt.title(f'Cumulative Returns for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    os.makedirs('output', exist_ok=True)
    plt.savefig(f'output/{symbol}_returns.png')
    plt.close()