import numpy as np

def psr(sharpe, T, skew, kurtosis):
    return sharpe * np.sqrt(T) / np.sqrt(1 - skew * sharpe + (kurtosis/4) * sharpe**2)

def dsr(sharpe, sharpe_std, N, T, skew, kurtosis):
    return (sharpe * np.sqrt(T) - N * sharpe_std) / np.sqrt(1 - skew * sharpe + (kurtosis/4) * sharpe**2)