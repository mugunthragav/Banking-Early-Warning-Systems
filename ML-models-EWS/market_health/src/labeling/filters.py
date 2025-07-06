import numpy as np
import pandas as pd

def cusum(close, h):
    t_events = []
    s_pos = s_neg = 0
    close_diff = np.log(close).diff()
    for i in range(1, len(close_diff)):
        s_pos = max(0, s_pos + close_diff.iloc[i])
        s_neg = min(0, s_neg + close_diff.iloc[i])
        if s_pos > h or s_neg < -h:
            t_events.append(close.index[i])
            s_pos = 0
            s_neg = 0
    return pd.DatetimeIndex(t_events)
