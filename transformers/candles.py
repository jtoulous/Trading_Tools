import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class EngulfingCandles(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = 0
        for idx in range(1, len(X)):
            prev_open = X.loc[idx-1, 'open']
            prev_close = X.loc[idx-1, 'close']
            prev_body = abs(prev_open - prev_close)

            open_price = X.loc[idx, 'open']
            close_price = X.loc[idx, 'close']
            body = abs(open_price - close_price)

            if close_price > open_price and prev_open > prev_close:  # Bullish engulfing
                if open_price <= prev_close and close_price >= prev_open:
                    ratio = body / (prev_body + 1e-6)
                    if ratio >= 2:
                        X.loc[idx, self.col_name] = 3
                    elif ratio >= 1.5:
                        X.loc[idx, self.col_name] = 2
                    else:
                        X.loc[idx, self.col_name] = 1
            elif open_price > close_price and prev_close > prev_open:  # Bearish engulfing
                if open_price >= prev_close and close_price <= prev_open:
                    ratio = body / (prev_body + 1e-6)
                    if ratio >= 2:
                        X.loc[idx, self.col_name] = -3
                    elif ratio >= 1.5:
                        X.loc[idx, self.col_name] = -2
                    else:
                        X.loc[idx, self.col_name] = -1
        return X



# PinBars : Identification de formations de pin bars
class PinBars(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = 0
        for idx, row in X.iterrows():
            open_price  = row['open']
            close_price = row['close']
            low_price   = row['low']
            high_price  = row['high']

            body_size    = abs(open_price - close_price)
            upper_shadow = abs(high_price - max(open_price, close_price))
            lower_shadow = abs(min(open_price, close_price) - low_price)

            if lower_shadow >= 4 * body_size:
                X.loc[idx, self.col_name] = 3
            elif lower_shadow >= 3 * body_size:
                X.loc[idx, self.col_name] = 2
            elif lower_shadow >= 2 * body_size:
                X.loc[idx, self.col_name] = 1
            elif upper_shadow >= 4 * body_size:
                X.loc[idx, self.col_name] = -3
            elif upper_shadow >= 3 * body_size:
                X.loc[idx, self.col_name] = -2
            elif upper_shadow >= 2 * body_size:
                X.loc[idx, self.col_name] = -1
        return X


class DojiCandles(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = 0
        for idx, row in X.iterrows():
            body_size  = abs(row['open'] - row['close'])
            range_size = row['high'] - row['low']
            ratio      = body_size / (range_size + 1e-6)

            if ratio < 0.05:
                X.loc[idx, self.col_name] = 3
            elif ratio < 0.1:
                X.loc[idx, self.col_name] = 2
            elif ratio < 0.2:
                X.loc[idx, self.col_name] = 1
        return X


class MarubozuCandles(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = 0
        for idx, row in X.iterrows():
            op, cl = row['open'], row['close']
            hi, lo  = row['high'], row['low']

            upper_shadow = hi - max(op, cl)
            lower_shadow = min(op, cl) - lo
            body         = abs(op - cl)
            total_shadow = upper_shadow + lower_shadow
            ratio        = total_shadow / (body + 1e-6)

            if ratio < 0.1:
                X.loc[idx, self.col_name] = 3
            elif ratio < 0.2:
                X.loc[idx, self.col_name] = 2
            elif ratio < 0.3:
                X.loc[idx, self.col_name] = 1
        return X


class ThreeWhiteSoldiers(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = 0
        for i in range(2, len(X)):
            o0, c0 = X.loc[i-2, 'open'], X.loc[i-2, 'close']
            o1, c1 = X.loc[i-1, 'open'], X.loc[i-1, 'close']
            o2, c2 = X.loc[i,   'open'], X.loc[i,   'close']
            if c0 > o0 and c1 > o1 and c2 > o2 and c2 > c1 and c1 > c0:
                X.loc[i, self.col_name] = 1
        return X


class ThreeBlackCrows(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = 0
        for i in range(2, len(X)):
            o0, c0 = X.loc[i-2, 'open'], X.loc[i-2, 'close']
            o1, c1 = X.loc[i-1, 'open'], X.loc[i-1, 'close']
            o2, c2 = X.loc[i,   'open'], X.loc[i,   'close']
            if c0 < o0 and c1 < o1 and c2 < o2 and c2 < c1 and c1 < c0:
                X.loc[i, self.col_name] = 1
        return X