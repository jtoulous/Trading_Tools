import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import hilbert

# Transformer pour extraire des features de date
class Date(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['DAY'] = X['timestamp'].dt.dayofweek.astype(float) + 1
        X['MONTH'] = X['timestamp'].dt.month.astype(float)
        X['YEAR'] = X['timestamp'].dt.year.astype(float)
        return X


class TR(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name
        pass
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = np.nan
        
        prev_row = None
        for idx, row in X.iterrows():
            if prev_row is not None:
                X.loc[idx, self.col_name] = max(
                    row['high'] - row['low'], 
                    abs(row['high'] - prev_row['close']), 
                    abs(row['low'] - prev_row['close'])
                )
            prev_row = row
        return X


class ATR(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        tr_trans = TR('TR_tmp')
        
        df = tr_trans.transform(df)
        df[self.col_name] = df['TR_tmp'].rolling(window=self.periods).mean()
        return X.assign(**{self.col_name: df[self.col_name]})


class ADX(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        
        tr_trans = TR('TR_tmp')
        df = tr_trans.transform(df)
       
        df['plus_dm'] = df['high'].diff()
        df['minus_dm'] = -df['low'].diff()
        
        df['plus_dm'] = df['plus_dm'].where((df['plus_dm'] > 0) & (df['plus_dm'] > df['minus_dm']), 0)
        df['minus_dm'] = df['minus_dm'].where((df['minus_dm'] > 0) & (df['minus_dm'] > df['plus_dm']), 0)
              
        df['tr_avg'] = df['TR_tmp'].rolling(window=self.periods).mean()
        df['plus_dm_avg'] = df['plus_dm'].rolling(window=self.periods).mean()
        df['minus_dm_avg'] = df['minus_dm'].rolling(window=self.periods).mean()

        df['plus_di'] = 100 * df['plus_dm_avg'] / df['tr_avg']
        df['minus_di'] = 100 * df['minus_dm_avg'] / df['tr_avg']

        df['dx'] = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
        df[self.col_name] = df['dx'].rolling(window=self.periods).mean()

        return X.assign(**{self.col_name: df[self.col_name]})


##############################
######        ICI


# EMA : Exponential Moving Average
class EMA(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):    
        X[self.col_name] = X['close'].ewm(span=self.periods, adjust=False).mean()
        return X



class Lagged(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, col_to_shift, shift_val=1):
        self.col_name = col_name
        self.col_to_shift = col_to_shift
        self.shift_val = shift_val

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = X[self.col_to_shift].shift(self.shift_val)
        return X



class RSI(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        delta = X['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.periods).mean()
        avg_loss = loss.rolling(window=self.periods).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (100 / (1 + rs))
        X[self.col_name] = rsi
        
        return X



class STO(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods  # (k_period, d_period)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        k_periods, d_periods = self.periods

        low_min = X['low'].rolling(window=k_periods).min()
        high_max = X['high'].rolling(window=k_periods).max()

        K = ((X['close'] - low_min) / (high_max - low_min)) * 100
        D = K.rolling(window=d_periods).mean()

        X[f'{self.col_name}_K'] = K
        X[f'{self.col_name}_D'] = D

        return X



# SMA : Simple Moving Average
class SMA(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = X['close'].rolling(window=self.periods).mean()
        return X



class WMA(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        weights = np.arange(1, self.periods + 1)
        weights = weights / weights.sum()

        X[self.col_name] = X['close'].rolling(window=self.periods).apply(lambda x: np.dot(x, weights), raw=True)
        return X


class DMI(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        up_move = X['high'].diff()
        down_move = -X['low'].diff()

        plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0)
        minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0)

        X[f'{self.col_name}_DM+'] = plus_dm
        X[f'{self.col_name}_DM-'] = minus_dm

        return X



class BLG(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods, num_std_dev):
        self.col_name = col_name
        self.periods = periods
        self.num_std_dev = num_std_dev

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        sma = X['close'].rolling(window=self.periods).mean()
        std = X['close'].rolling(window=self.periods).std()

        upper_band = sma + self.num_std_dev * std
        lower_band = sma - self.num_std_dev * std
        band_width = upper_band - lower_band

        X[f'{self.col_name}_upper'] = upper_band
        X[f'{self.col_name}_lower'] = lower_band
        X[f'{self.col_name}_width'] = band_width

        return X


class MACD(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, short, long, signal):
        self.col_name = col_name
        self.short_period = short
        self.long_period = long
        self.signal_period = signal

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ema_short = X['close'].ewm(span=self.short_period, adjust=False).mean()
        ema_long = X['close'].ewm(span=self.long_period, adjust=False).mean()
        macd_line = ema_short - ema_long
        macd_signal = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        macd_histo = macd_line - macd_signal

        X[f'{self.col_name}_line'] = macd_line
        X[f'{self.col_name}_signal'] = macd_signal
        X[f'{self.col_name}_histo'] = macd_histo

        return X


# HilbertsTransform : utilise la transformée de Hilbert
class HilbertsTransform(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analytic_signal = hilbert(X['close'])
        X[self.col_name] = analytic_signal.imag
        return X


class CCI(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, periods):
        self.col_name = col_name
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tp = (X['high'] + X['low'] + X['close']) / 3  # Typical Price
        sma_tp = tp.rolling(window=self.periods).mean()
        md = tp.rolling(window=self.periods).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - sma_tp) / (0.015 * md)
        X[self.col_name] = cci
        return X


# PPO : Percentage Price Oscillator
class PPO(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, short, long, signal):
        self.col_name = col_name
        self.short = short
        self.long = long
        self.signal = signal

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        short_ema = X['close'].ewm(span=self.short).mean()
        long_ema = X['close'].ewm(span=self.long).mean()
        ppo_line = ((short_ema - long_ema) / long_ema) * 100
        signal_line = ppo_line.ewm(span=self.signal).mean()

        X[f'{self.col_name}_line'] = ppo_line
        X[f'{self.col_name}_signal'] = signal_line
        return X


# ROC : Rate of Change
class ROC(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.col_name] = X['close'].pct_change()
        return X