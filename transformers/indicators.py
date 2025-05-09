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
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['TR'] = np.nan
        
        prev_row = None
        for idx, row in X.iterrows():
            if prev_row is not None:
                X.loc[idx, 'TR'] = max(
                    row['high'] - row['low'], 
                    abs(row['high'] - prev_row['close']), 
                    abs(row['low'] - prev_row['close'])
                )
            prev_row = row
        return X


class ATR(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if 'TR' not in X.columns:    
            tr_transformer = TR()
            X = tr_transformer.transform(X)
        
        X[f'ATR_{self.periods}'] = X['TR'].rolling(window=self.periods).mean()
        return X


class ADX(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Si TR pas déjà présent, on le calcule avec le transformer dédié
        if 'TR' not in X.columns:
            tr_transformer = TR()
            X = tr_transformer.transform(X)

        # +DM / -DM
        plus_dm = X['high'].diff()
        minus_dm = -X['low'].diff()

        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

        # Moyennes
        tr_avg = X['TR'].rolling(window=self.periods).mean()
        plus_dm_avg = plus_dm.rolling(window=self.periods).mean()
        minus_dm_avg = minus_dm.rolling(window=self.periods).mean()

        plus_di = 100 * plus_dm_avg / tr_avg
        minus_di = 100 * minus_dm_avg / tr_avg

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        X[f'ADX_{self.periods}'] = dx.rolling(window=self.periods).mean()

        return X



# EMA : Exponential Moving Average
class EMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):    
        X[f'EMA_{self.periods}'] = X['close'].ewm(span=self.periods, adjust=False).mean()
        return X



class Lagged(BaseEstimator, TransformerMixin):
    def __init__(self, columns, shift_val=1):
        self.columns = columns
        self.shift_val = shift_val

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in self.columns:
            X[f'{column}_lag_{self.shift_val}'] = X[column].shift(self.shift_val)
        return X



class RSI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
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
        X[f'RSI_{self.periods}'] = rsi
        
        return X



class STO(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods  # (k_period, d_period)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        k_periods, d_periods = self.periods

        low_min = X['low'].rolling(window=k_periods).min()
        high_max = X['high'].rolling(window=k_periods).max()

        K = ((X['close'] - low_min) / (high_max - low_min)) * 100
        D = K.rolling(window=d_periods).mean()

        X['K(sto)'] = K
        X['D(sto)'] = D

        return X



# SMA : Simple Moving Average
class SMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['SMA'] = X['close'].rolling(window=self.periods).mean()
        return X



class WMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        weights = np.arange(1, self.periods + 1)
        weights = weights / weights.sum()

        X['WMA'] = X['close'].rolling(window=self.periods).apply(lambda x: np.dot(x, weights), raw=True)
        return X


class DMI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        up_move = X['high'].diff()
        down_move = -X['low'].diff()

        plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0)
        minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0)

        plus_dm_avg = plus_dm.rolling(window=self.periods).mean()
        minus_dm_avg = minus_dm.rolling(window=self.periods).mean()

        dm_diff = (plus_dm_avg - minus_dm_avg).abs()
        dm_sum = plus_dm_avg + minus_dm_avg

        dx = 100 * dm_diff / dm_sum.replace(0, np.nan)
        X[f'ADX_{self.periods}'] = dx.rolling(window=self.periods).mean()

        X[f'DM+_{self.periods}'] = plus_dm
        X[f'DM-_{self.periods}'] = minus_dm

        return X



class BLG(BaseEstimator, TransformerMixin):
    def __init__(self, periods, num_std_dev):
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

        X[f'BLG_UPPER_{self.periods}'] = upper_band
        X[f'BLG_LOWER_{self.periods}'] = lower_band
        X[f'BLG_WIDTH_{self.periods}'] = band_width

        return X


class MACD(BaseEstimator, TransformerMixin):
    def __init__(self, short, long, signal):
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

        X[f'MACD_LINE_{self.short_period}_{self.long_period}_{self.signal_period}'] = macd_line
        X[f'MACD_SIGNAL_{self.short_period}_{self.long_period}_{self.signal_period}'] = macd_signal
        X[f'MACD_HISTO_{self.short_period}_{self.long_period}_{self.signal_period}'] = macd_histo

        return X


# HilbertsTransform : utilise la transformée de Hilbert
class HilbertsTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analytic_signal = hilbert(X['close'])
        X['HBT_TRANS'] = analytic_signal.imag
        return X


class CCI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tp = (X['high'] + X['low'] + X['close']) / 3  # Typical Price
        sma_tp = tp.rolling(window=self.periods).mean()
        md = tp.rolling(window=self.periods).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - sma_tp) / (0.015 * md)
        X[f'CCI_{self.periods}'] = cci
        return X


# PPO : Percentage Price Oscillator
class PPO(BaseEstimator, TransformerMixin):
    def __init__(self, short, long, signal):
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

        X[f'PPO_LINE_{self.short}_{self.long}_{self.signal}'] = ppo_line
        X[f'PPO_SIGNAL_{self.short}_{self.long}_{self.signal}'] = signal_line
        return X


# ROC : Rate of Change
class ROC(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['ROC'] = X['close'].pct_change()
        return X


# FuturFluctuation : Calcul du score de fluctuation future
class FuturFluctuation(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        scores = []
        closes = X['close']
        n = len(X)
        for i in range(n):
            current = closes.iloc[i]
            future_window = closes.iloc[i+1: i+1+self.periods]
            if len(future_window) == 0:
                scores.append(np.nan)
                continue

            up_pct = (future_window.max() - current) / current * 100
            down_pct = (current - future_window.min()) / current * 100

            score = up_pct if up_pct >= down_pct else -down_pct
            score = max(-10, min(10, score))
            scores.append(score)

        X['futur_fluctuation'] = scores
        return X



class PastVolatilityPercentile(BaseEstimator, TransformerMixin):
    def __init__(self, window):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        returns = X['close'].pct_change()

        vol = returns.rolling(window=self.window).std()

        def pct_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        X[f'vol_past_{self.window}_pct'] = vol.rolling(window=self.window).apply(pct_rank, raw=False)
        return X


class FutureVolatilityPercentile(BaseEstimator, TransformerMixin):
    def __init__(self, window: int):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        returns = X['close'].pct_change()

        # On renverse pour regarder dans le futur
        vol_rev = returns[::-1].rolling(window=self.window).std()

        def pct_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1]

        score_rev = vol_rev.rolling(window=self.window).apply(pct_rank, raw=False)

        # On remet dans l’ordre
        X[f'vol_future_{self.window}_pct'] = score_rev[::-1]
        return X


class PastVolatilityPercentile(BaseEstimator, TransformerMixin):
    def __init__(self, window: int):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        returns = X['close'].pct_change()  
        vol = returns.rolling(window=self.window).std()  
        scores = []
        for i in range(len(vol)):
            if i < self.window - 1:
                scores.append(float('nan'))
                continue
            window_vals = vol.iloc[i-self.window+1 : i+1].values
            rank = (window_vals <= vol.iat[i]).sum()
            pct = rank / self.window
            scores.append(pct)

        X[f'vol_past_{self.window}_pct'] = scores
        return X


class FutureVolatilityPercentile(BaseEstimator, TransformerMixin):
    def __init__(self, window):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        returns = X['close'].pct_change()
        vol = returns[::-1].rolling(window=self.window).std()[::-1]

        scores = []
        for i in range(len(vol)):
            if i > len(vol) - self.window:
                scores.append(float('nan'))
                continue
            window_vals = vol.iloc[i : i+self.window].values
            rank = (window_vals <= vol.iat[i]).sum()
            pct = rank / self.window
            scores.append(pct)

        X[f'vol_future_{self.window}_pct'] = scores
        return X
