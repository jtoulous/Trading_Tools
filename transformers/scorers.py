import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .indicators import ATR, ADX



class VolatilityScorer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, periods, shift=False):
        self.column_name = column_name
        self.periods = periods
        self.shift = shift

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        atr_transformer = ATR('ATR_tmp', self.periods)
            
        df = atr_transformer.transform(df)

        df['ATR_zscore'] = (df['ATR_tmp'] - df['ATR_tmp'].mean()) / df['ATR_tmp'].std()
        
        if self.shift is False:
            df[self.column_name] = np.tanh(df['ATR_zscore'])
        else:
            df['Volatility_score_tmp'] =  np.tanh(df['ATR_zscore'])
            df[self.column_name] = df['Volatility_score_tmp'].shift(self.periods)

        return X.assign(**{self.column_name: df[self.column_name]})



class TrendDirectionPowerScorer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, periods, shift=False):
        self.column_name = column_name
        self.periods = periods
        self.shift = shift

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        df['Trend_diff_tmp'] = df['close'].diff()
        df['Trend_direction_sum_tmp'] = df['Trend_diff_tmp'].rolling(window=self.periods).sum()
        df['Trend_direction_zscore_tmp'] = (df['Trend_direction_sum_tmp'] - df['Trend_direction_sum_tmp'].mean()) / df['Trend_direction_sum_tmp'].std()
        
        if self.shift is False:
            df[self.column_name] = np.tanh(df['Trend_direction_zscore_tmp'])
        else:
            df['Trend_direction_score_tmp'] = np.tanh(df['Trend_direction_zscore_tmp'])
            df[self.column_name] = df['Trend_direction_score_tmp'].shift(self.periods)

        return X.assign(**{self.column_name: df[self.column_name]})



class TrendPersistanceScorer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, periods, shift=False):
        self.column_name = column_name
        self.periods = periods
        self.shift = shift

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        df['delta_tmp'] = df['close'].diff().fillna(0)
        sign = np.sign(df['delta_tmp'])

        sign = sign.replace(0, np.nan).ffill().fillna(0)

        new_group = (sign != sign.shift(1)).cumsum()
        run_length = new_group.groupby(new_group).cumcount() + 1

        bullish_power = (run_length * df['delta_tmp'].clip(lower=0)).rolling(self.periods).sum()
        bearish_power = (run_length * (-df['delta_tmp'].clip(upper=0))).rolling(self.periods).sum()

        persistence = (bullish_power - bearish_power) / (bullish_power + bearish_power).replace(0, np.nan)

        if self.shift:
            persistence = persistence.shift(self.periods)

        return X.assign(**{self.column_name: persistence})