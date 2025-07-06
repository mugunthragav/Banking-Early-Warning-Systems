import numpy as np
import pandas as pd

def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads=4):
    events = pd.DataFrame(index=tEvents)
    events['t1'] = events.index + pd.offsets.Minute(1)
    events['trgt'] = trgt
    events['pt'] = ptSl[0] * trgt
    events['sl'] = -ptSl[1] * trgt
    if minRet > 0: events = events[close.loc[events.index].pct_change() >= minRet]
    return events

def getBins(events, close):
    label = pd.Series(0, index=close.index)
    for t1 in events['t1']:
        if t1 in close.index:
            label.loc[t1] = 1
    return pd.DataFrame({'bin': label})