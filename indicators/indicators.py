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

s        rs = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (100 / (1 + rs))
        X[f'RSI_{self.periods}'] = rsi
        
        return X





######################################################
#              ICIIIII
######################################################


# STO : Stochastic Oscillator
class STO(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods  # periods is a tuple (k_period, d_period)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        k_periods = self.periods[0]
        d_periods = self.periods[1]

        low_min = X['low'].rolling(window=k_periods, min_periods=1).min()
        high_max = X['high'].rolling(window=k_periods, min_periods=1).max()
        K = ((X['close'] - low_min) / (high_max - low_min)) * 100
        D = K.rolling(window=d_periods, min_periods=1).mean()

        X['K(sto)'] = K
        X['D(sto)'] = D
        return X.dropna().reset_index(drop=True)


# SMA : Simple Moving Average
class SMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['SMA'] = X['close'].rolling(window=self.periods, min_periods=1).mean()
        return X.dropna().reset_index(drop=True)



# WMA : Weighted Moving Average
class WMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        weights = pd.Series(np.arange(1, self.periods + 1) / np.sum(np.arange(1, self.periods + 1)))
        weighted_close = X['close'].rolling(window=self.periods).apply(lambda x: (x * weights).sum(), raw=True)
        X['WMA'] = weighted_close
        return X.dropna().reset_index(drop=True)


# DMI : Directional Movement Index
class DMI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        prev_row = None
        dm_plus = []
        dm_minus = []
        for idx, row in X.iterrows():
            if prev_row is not None:
                dm_plus_val = row['high'] - prev_row['high'] if (row['high'] - prev_row['high'] > prev_row['low'] - row['low'] and row['high'] - prev_row['high'] > 0) else 0
                dm_minus_val = prev_row['low'] - row['low'] if (prev_row['low'] - row['low'] > row['high'] - prev_row['high'] and prev_row['low'] - row['low'] > 0) else 0
            else:
                dm_plus_val = 0
                dm_minus_val = 0
            dm_plus.append(dm_plus_val)
            dm_minus.append(dm_minus_val)
            prev_row = row

        dm_plus = pd.Series(dm_plus)
        dm_minus = pd.Series(dm_minus)
        dm_diff = abs(dm_plus - dm_minus)
        dm_sum = dm_plus + dm_minus
        dx = dm_diff / dm_sum.replace(0, np.nan)
        adx = dx.rolling(window=self.periods, min_periods=1).mean()

        X['DM+'] = dm_plus
        X['DM-'] = dm_minus
        X['ADX'] = adx
        return X.dropna().reset_index(drop=True)


# BLG : Bollinger Bands-like Indicator
class BLG(BaseEstimator, TransformerMixin):
    def __init__(self, periods, num_std_dev, live=False):
        self.periods = periods
        self.num_std_dev = num_std_dev
        self.live = live

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if not self.live:
            TMP_SMA = X['close'].rolling(window=self.periods, min_periods=1).mean()
            TMP_STD = X['close'].rolling(window=self.periods, min_periods=1).std()
            u_band = TMP_SMA + (TMP_STD * self.num_std_dev)
            l_band = TMP_SMA - (TMP_STD * self.num_std_dev)
            X['U-BAND'] = u_band
            X['L-BAND'] = l_band
            X['BLG_WIDTH'] = X['U-BAND'] - X['L-BAND']
        else:
            for i in range(1, len(X)):
                if pd.isna(X.loc[i, 'U-BAND']) or pd.isna(X.loc[i, 'L-BAND']):
                    TMP_SMA = X['close'].iloc[max(i - self.periods + 1, 0):i+1].mean()
                    TMP_STD = X['close'].iloc[max(i - self.periods + 1, 0):i+1].std()
                    u_band = TMP_SMA + (TMP_STD * self.num_std_dev)
                    l_band = TMP_SMA - (TMP_STD * self.num_std_dev)
                    X.loc[i, 'U-BAND'] = u_band
                    X.loc[i, 'L-BAND'] = l_band
                    X.loc[i, 'BLG_WIDTH'] = u_band - l_band
        return X.dropna().reset_index(drop=True)


# MACD : Moving Average Convergence Divergence
class MACD(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        short_period = self.args[0]
        long_period = self.args[1]
        signal_period = self.args[2]

        ema_short = X['close'].rolling(window=short_period, min_periods=1).mean()
        ema_long = X['close'].rolling(window=long_period, min_periods=1).mean()
        macd_line = ema_short - ema_long
        macd_signal = macd_line.ewm(span=signal_period, min_periods=1).mean()
        macd_histo = macd_line - macd_signal

        X['MACD_LINE'] = macd_line
        X['MACD_SIGNAL'] = macd_signal
        X['MACD_HISTO'] = macd_histo
        return X.dropna().reset_index(drop=True)


# HilbertsTransform : utilise la transformée de Hilbert
class HilbertsTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        analytic_signal = hilbert(X['close'])
        X['HBT_TRANS'] = analytic_signal.imag
        return X.dropna().reset_index(drop=True)


# CCI : Commodity Channel Index
class CCI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        TP = (X['high'] + X['low'] + X['close']) / 3
        SMA_TP = TP.rolling(window=self.periods, min_periods=1).mean()
        MD = TP.rolling(window=self.periods, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        X['CCI'] = (TP - SMA_TP) / (0.015 * MD)
        return X.dropna().reset_index(drop=True)


# PPO : Percentage Price Oscillator
class PPO(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        short_period = self.periods[0]
        long_period = self.periods[1]
        signal_period = self.periods[2]

        short_ema = X['close'].ewm(span=short_period, min_periods=1).mean()
        long_ema = X['close'].ewm(span=long_period, min_periods=1).mean()
        ppo_line = ((short_ema - long_ema) / long_ema) * 100
        signal_line = ppo_line.ewm(span=signal_period, min_periods=1).mean()

        X['PPO_LINE'] = ppo_line
        X['PPO_SIGNAL'] = signal_line
        return X.dropna().reset_index(drop=True)


# ROC : Rate of Change
class ROC(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['ROC'] = X['close'].pct_change()
        return X.dropna().reset_index(drop=True)


# Slope : Calcul de la pente sur une fenêtre donnée
class Slope(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        slopes = []
        for i in range(self.periods, len(X)):
            y_vals = X['close'].iloc[i - self.periods:i]
            x_vals = np.arange(self.periods)
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            slopes.append(slope)
        slopes = [np.nan] * (self.periods) + slopes
        X['SLOPE'] = slopes
        return X.dropna().reset_index(drop=True)


# Z_SCORE : Calcul du Z-Score pour une colonne donnée
class Z_SCORE(BaseEstimator, TransformerMixin):
    def __init__(self, periods, column):
        self.periods = periods
        self.column = column
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        col_mean = X[self.column].rolling(window=self.periods, min_periods=1).mean()
        col_std = X[self.column].rolling(window=self.periods, min_periods=1).std()
        X[f'Z_SCORE_{self.column}'] = (X[self.column] - col_mean) / col_std
        return X.dropna().reset_index(drop=True)


# Growth : Calcul de la croissance (méthode de ton choix)
class Growth(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['GROWTH'] = (X['close'] - X['volume']) / X['volume'] * 100
        return X.dropna().reset_index(drop=True)


# HV : Historical Volatility
class HV(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['log_returns'] = np.log(X['close'] / X['close'].shift(1))
        X[f'HV{self.periods}'] = X['log_returns'].rolling(window=self.periods).std()
        X.drop(columns=['log_returns'], inplace=True)
        return X.dropna().reset_index(drop=True)


# OBV : On Balance Volume
class OBV(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['OBV'] = np.nan
        for i in range(1, len(X)):
            if X['close'].iloc[i] > X['close'].iloc[i - 1]:
                X.loc[i, 'OBV'] = X['volume'].iloc[i]
            elif X['close'].iloc[i] < X['close'].iloc[i - 1]:
                X.loc[i, 'OBV'] = -X['volume'].iloc[i]
            else:
                X.loc[i, 'OBV'] = 0
        X['OBV'] = X['OBV'].cumsum()
        return X.dropna().reset_index(drop=True)


# CMF : Chaikin Money Flow
class CMF(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['CMF'] = np.nan
        for i in range(len(X)):
            if X['high'].iloc[i] != X['low'].iloc[i]:
                money_flow_multiplier = ((X['close'].iloc[i] - X['low'].iloc[i]) - (X['high'].iloc[i] - X['close'].iloc[i])) / (X['high'].iloc[i] - X['low'].iloc[i])
                money_flow_volume = money_flow_multiplier * X['volume'].iloc[i]
                X.loc[i, 'CMF'] = money_flow_volume
        X['CMF'] = X['CMF'].rolling(window=self.periods, min_periods=1).sum() / X['volume'].rolling(window=self.periods, min_periods=1).sum()
        return X.dropna().reset_index(drop=True)


# RealizedVolatility : Volatilité réalisée
class RealizedVolatility(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['log_returns'] = np.log(X['close'] / X['close'].shift(1))
        X[f'VOLATILITY_{self.periods}'] = X['log_returns'].rolling(window=self.periods).std() * np.sqrt(self.periods)
        X.drop(columns=['log_returns'], inplace=True)
        return X.dropna().reset_index(drop=True)


# PinBars : Identification de formations de pin bars
class PinBars(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for idx, row in X.iterrows():
            open_price = row['open']
            close_price = row['close']
            low_price = row['low']
            high_price = row['high']

            body_size = abs(open_price - close_price)
            upper_shadow = abs(high_price - open_price) if open_price > close_price else abs(high_price - close_price)
            lower_shadow = abs(low_price - close_price) if open_price > close_price else abs(low_price - high_price)

            if lower_shadow >= 4 * body_size:
                X.loc[idx, 'pin_bar'] = 3
            elif lower_shadow >= 3 * body_size:
                X.loc[idx, 'pin_bar'] = 2
            elif lower_shadow >= 2 * body_size:
                X.loc[idx, 'pin_bar'] = 1
            elif upper_shadow >= 4 * body_size:
                X.loc[idx, 'pin_bar'] = -3
            elif upper_shadow >= 3 * body_size:
                X.loc[idx, 'pin_bar'] = -2
            elif upper_shadow >= 2 * body_size:
                X.loc[idx, 'pin_bar'] = -1
            else:
                X.loc[idx, 'pin_bar'] = 0
        return X.dropna().reset_index(drop=True)


# FuturFluctuation : Calcul du score de fluctuation future
class FuturFluctuation(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
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
        return X.dropna().reset_index(drop=True)
