import numpy as np
from multiprocessing import Pool
import pandas as pd


def process_chunk(chunk):
    """Process a chunk of signals to calculate average active signals."""
    active = np.nansum(chunk, axis=0) / chunk.shape[0]
    return active


def avgActiveSignals(signals, window=50):
    """Calculate the average active signals across a rolling window using multiprocessing."""
    signals_array = signals.values
    n = len(signals)
    chunks = [signals_array[max(0, i - window):i + 1] for i in range(window, n)]

    with Pool() as pool:
        results = pool.map(process_chunk, chunks)

    avg_signals = pd.Series(np.nan, index=signals.index)
    for i, active in enumerate(results, start=window):
        avg_signals.iloc[i] = active if not np.isnan(active) else np.nan

    return avg_signals