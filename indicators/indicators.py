import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import hilbert


class DateToFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['DAY'] = X['timestamp'].dt.dayofweek + 1
        X['DAY'] = X['DAY'].astype(float)
        return X


class TR(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        prev_row = None

        for idx, row in X.iterrows():
            if idx == 0:
                X.loc[idx, 'TR'] = row['high'] - row['low']

            else:
                X.loc[idx, 'TR'] = max(row['high'] - row['low'], abs(row['high'] - prev_row['close']), abs(row['low'] - prev_row['close']))
            prev_row = row
        return X


class PriceRange(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['PR'] = X['high'] - X['low']
        return X


class ATR(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods
 
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tr = []
        prev_row = None

        for idx, row in X.iterrows():
            if idx == 0:
                tr_val = row['high'] - row['low']
            else:
                tr_val = max(row['high'] - row['low'], abs(row['high'] - prev_row['close']), abs(row['low'] - prev_row['close']))
            tr.append(tr_val)
            prev_row = row

        tr = pd.Series(tr)
        X[f'ATR{str(self.periods)}'] = tr.rolling(window=self.periods, min_periods=1).mean()
        return X


class ADX():
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['H-L_TMP'] = X['high'] - X['low']
        X['H-Close_TMP'] = abs(X['high'] - X['close'].shift())
        X['L-Close_TMP'] = abs(X['low'] - X['close'].shift())
        X['TR_TMP'] = X[['H-L_TMP', 'H-Close_TMP', 'L-Close_TMP']].max(axis=1)

        X['+DM_TMP'] = X['high'].diff()
        X['-DM_TMP'] = -X['low'].diff()

        X['+DM_TMP'] = X['+DM_TMP'].where(X['+DM_TMP'] > 0, 0)
        X['-DM_TMP'] = X['-DM_TMP'].where(X['-DM_TMP'] > 0, 0)

        X['+DM_avg_TMP'] = X['+DM_TMP'].rolling(window=self.periods).mean()
        X['-DM_avg_TMP'] = X['-DM_TMP'].rolling(window=self.periods).mean()
        X['TR_avg_TMP'] = X['TR_TMP'].rolling(window=self.periods).mean()

        X['+DI'] = 100 * X['+DM_avg_TMP'] / X['TR_avg_TMP']
        X['-DI'] = 100 * X['-DM_avg_TMP'] / X['TR_avg_TMP']

        X[f'ADX{self.periods}'] = 100 * (abs(X['+DI'] - X['-DI']) / (X['+DI'] + X['-DI'])).rolling(window=self.periods).mean()

        X.drop(columns=['H-L_TMP', 'H-Close_TMP', 'L-Close_TMP', 'TR_TMP', '+DM_TMP', '-DM_TMP', '+DM_avg_TMP', '-DM_avg_TMP', 'TR_avg_TMP', '+DI', '-DI'], inplace=True)
        return X     



class EMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):    
        X[f'EMA{str(self.periods)}'] = X['close'].ewm(span=self.periods, adjust=False).mean()
        return X


class EMA_Lagged(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.ema_periods = periods 
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'EMA{str(self.ema_periods)}_Lagged'] = X[f'EMA{self.ema_periods}'].shift(1).bfill().ffill()
        return X


class EMA_SMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.ema_periods = periods
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'EMA{str(self.ema_periods)}_SMA_Ratio'] = X[f'EMA{self.ema_periods}'] / X['SMA'].bfill().ffill()
        return X


class RSI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        delta = X['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (delta.where(delta < 0, 0)).fillna(0)
        loss = -loss

        avg_gain = gain.rolling(window=self.periods, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.periods, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        X[f'RSI{str(self.periods)}'] = rsi.bfill()
        return X


class STO(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        k_periods = self.periods[0]
        d_periods = self.periods[1]

        low_min = X['low'].rolling(window=k_periods, min_periods=1).min()
        high_max = X['high'].rolling(window=k_periods,min_periods=1).max()
        K = ((X['close'] - low_min) / (high_max - low_min)) * 100
        D = K.rolling(window=d_periods, min_periods=1).mean()

        X['K(sto)'] = K.bfill()
        X['D(sto)'] = D.bfill()
        return X
    

class SMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['SMA'] = X['close'].rolling(window=self.periods, min_periods=1).mean()
        X['SMA'] = X['SMA'].bfill()
        return X


class SMA_DIFF(BaseEstimator, TransformerMixin):
    def __init__(self, period_1, period_2):
        self.period_1 = period_1
        self.period_2 = period_2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['tmp_sma_1'] = X['close'].rolling(window=self.period_1, min_periods=1).mean()
        X['tmp_sma_2'] = X['close'].rolling(window=self.period_2, min_periods=1).mean()
        X[f'SMA_DIFF_{self.period_1}_{self.period_2}'] = X['tmp_sma_1'] - X['tmp_sma_2']
        X.drop(columns=['tmp_sma_1', 'tmp_sma_2'], inplace=True)
        return X



class WMA(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        weights = pd.Series(np.arange(1, self.periods + 1) / np.sum(np.arange(1, self.periods + 1)))
        weighted_close = X['close'].rolling(window=self.periods).apply(lambda x: (x * weights).sum(), raw=True)
        X['WMA'] = weighted_close.bfill()
        return X


class DMI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        prev_row = None
        dm_plus = []
        dm_minus = []

        for idx, row in X.iterrows():
            if prev_row is not None:
                dm_plus_val = row['high'] - prev_row['high'] \
                                if row['high'] - prev_row['high'] > prev_row['low'] - row['low'] \
                                and row['high'] - prev_row['high'] > 0 \
                                else 0
                dm_minus_val = prev_row['low'] - row['low'] \
                                if prev_row['low'] - row['low'] > row['high'] - prev_row['high'] \
                                and prev_row['low'] - row['low'] > 0 \
                                else 0
            else:
                dm_plus_val = 0
                dm_minus_val = 0
            dm_plus.append(dm_plus_val)
            dm_minus.append(dm_minus_val)
            prev_row = row

        dm_plus = pd.Series(dm_plus).bfill()
        dm_minus = pd.Series(dm_minus).bfill()
        dm_diff = abs(dm_plus - dm_minus)
        dm_sum = dm_plus + dm_minus
        dx = dm_diff / dm_sum
        adx = dx.rolling(window=self.periods, min_periods=1).mean().bfill()

        X['DM+'] = dm_plus
        X['DM-'] = dm_minus
        X['ADX'] = adx
        return X


class BLG(BaseEstimator, TransformerMixin):
    def __init__(self, periods, num_std_dev, live=False):
        self.periods = periods
        self.num_std_dev = num_std_dev
        self.live = live


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.live is False:
            TMP_SMA = X['close'].rolling(window=self.periods, min_periods=1).mean()
            TMP_STD = X['close'].rolling(window=self.periods, min_periods=1).std()
            u_band = TMP_SMA + (TMP_STD * self.num_std_dev)
            l_band = TMP_SMA - (TMP_STD * self.num_std_dev)

            X['U-BAND'] = u_band.bfill().ffill()
            X['L-BAND'] = l_band.bfill().ffill()
            X['BLG_WIDTH'] = (X['U-BAND'] - X['L-BAND'])

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

        return X


class MACD(BaseEstimator, TransformerMixin):
    def __init__(self, args):
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        short_period = self.args[0]
        long_period = self.args[1]
        signal_period = self.args[2]

        ema_short = X['close'].rolling(window=short_period, min_periods=1).mean()
        ema_long = X['close'].rolling(window=long_period, min_periods=1).mean()
        macd_line = ema_short - ema_long
        macd_signal = macd_line.ewm(span=signal_period, min_periods=1).mean()
        macd_histo = macd_line - macd_signal

        X['MACD_LINE'] = macd_line.bfill()
        X['MACD_SIGNAL'] = macd_signal.bfill()
        X['MACD_HISTO'] = macd_histo.bfill()
        return X


class HilbertsTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):    
        analytic_signal = hilbert(X['close'])
        X['HBT_TRANS'] = analytic_signal.imag
        X['HBT_TRANS'] = X['HBT_TRANS'].bfill()
        return X


class CCI(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        TP = (X['high'] + X['low'] + X['close']) / 3
        SMA_TP = TP.rolling(window=self.periods, min_periods=1).mean()
        MD = TP.rolling(window=self.periods, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        CCI = (TP -SMA_TP) / (0.015 * MD)
        X['CCI'] = CCI.bfill()
        return X


class PPO(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        short_period = self.periods[0]
        long_period = self.periods[1]
        signal_period = self.periods[2]

        short_ema = X['close'].ewm(span=short_period, min_periods=1).mean()
        long_ema = X['close'].ewm(span=long_period, min_periods=1).mean()
        ppo_line = ((short_ema - long_ema) / long_ema) * 100
        signal_line = ppo_line.ewm(span=signal_period, min_periods=1).mean()

        X['PPO_LINE'] = ppo_line.bfill().ffill()
        X['PPO_SIGNAL'] = signal_line.bfill().ffill()
        return X


class ROC(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['ROC'] = X['close'].pct_change()
        X['ROC'] = X['ROC'].bfill()
        return X



class Slope(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        slopes = []

        for i in range(self.periods, len(X)):
            y_vals = X['close'].iloc[i - self.periods:i]
            x_vals = range(self.periods)
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            slopes.append(slope)

        slopes = [np.nan] * (self.periods) + slopes
        X['SLOPE'] = slopes
        return X



class   Z_SCORE(BaseEstimator, TransformerMixin):
    def __init__(self, periods, column):
        self.periods = periods
        self.column = column
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        col_mean = X[self.column].rolling(window=self.periods, min_periods=1).mean()
        col_std = X[self.column].rolling(window=self.periods, min_periods=1).std()

        X[f'Z_SCORE_{self.column}'] = (X[self.column] - col_mean) / col_std
        return X


class   Growth(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['GROWTH'] = (X['close'] - X['volume']) / X['volume'] * 100
        return X


class HV(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['log_returns'] = np.log(X['close'] / X['close'].shift(1))
        X[f'HV{self.periods}'] = X['log_returns'].rolling(window=self.periods).std()
        
        X.drop(columns=['log_returns'], inplace=True)
        
        return X


class OBV(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['OBV'] = np.nan
        
        for i in range(1, len(X)):
            if X['close'].iloc[i] > X['close'].iloc[i - 1]:
                X.loc[i, 'OBV'] = X['volume'].iloc[i]
            elif X['close'].iloc[i] < X['close'].iloc[i - 1]:
                X.loc[i, 'OBV'] = -X['volume'].iloc[i]
            else:
                X.loc[i, 'OBV'] = 0

        X['OBV'] = X['OBV'].cumsum()
        X = X.dropna(subset=['OBV']).reset_index(drop=True)
        return X



class CMF(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['CMF'] = np.nan

        for i in range(len(X)):
            if X['high'].iloc[i] != X['low'].iloc[i]:
                money_flow_multiplier = ((X['close'].iloc[i] - X['low'].iloc[i]) - (X['high'].iloc[i] - X['close'].iloc[i])) / (X['high'].iloc[i] - X['low'].iloc[i])
                money_flow_volume = money_flow_multiplier * X['volume'].iloc[i]
                X.loc[i, 'CMF'] = money_flow_volume

        X['CMF'] = X['CMF'].rolling(window=self.periods, min_periods=1).sum() / X['volume'].rolling(window=self.periods, min_periods=1).sum()
        X = X.dropna(subset=['CMF']).reset_index(drop=True)

        return X


class AmplitudeMax(BaseEstimator, TransformerMixin):######  ICI
    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        return self

#    def transform(self, X, y=None):
#        X[f'AMP{self.period}'] = (
#            X['high'].rolling(window=self.period, min_periods=1).max() -
#            X['low'].rolling(window=self.period, min_periods=1).min()
#        )
#        for i in range(self.period + 1, len(X)):
#            max_high = X['high'].iloc[i]
#            min_low = X['low'].iloc[i]
#            for j in range(i - self.period, i):
#                
#        return X



class RealizedVolatility(BaseEstimator, TransformerMixin):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[f'VOLATILITY_{self.periods}'] = np.nan
        X[f'log_returns'] = np.nan
        
        X['log_returns'] = np.log(X['close'] / X['close'].shift(1))
        X[f'VOLATILITY_{self.periods}'] = X['log_returns'].rolling(window=self.periods).std() * np.sqrt(self.periods)
        X.drop(columns=['log_returns'], inplace=True)
        
        X = X.dropna(subset=[f'VOLATILITY_{self.periods}']).reset_index(drop=True)
        return X



class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['ATR_Lagged'] = X['ATR'].shift(1).bfill()
        X['EMA_Lagged'] = X['EMA'].shift(1).bfill()
    #    X['RSI_Lagged'] = X['RSI'].shift(1).bfill()
    #    X['Momentum'] = X['close'] - X['close'].shift(1).bfill()
        X['MACD_Difference'] = X['MACD_LINE'] - X['MACD_SIGNAL'].bfill()
        X['BLG_WIDTH'] = X['U-BAND'] - X['L-BAND'].bfill()
        X['EMA_SMA_Ratio'] = X['EMA'] / X['SMA'].bfill()
        return X