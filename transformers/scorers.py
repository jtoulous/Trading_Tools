import numpy as np
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
            df[self.column_name] = df['Volatility_score_tmp'].shift(-self.periods)

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
            df[self.column_name] = df['Trend_direction_score_tmp'].shift(-self.periods)

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
            persistence = persistence.shift(-self.periods)

        return X.assign(**{self.column_name: persistence})



class LongOrShort(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, time_to_live, multiplier, ratio):
        self.column_name = column_name
        self.ttl = time_to_live
        self.multiplier = multiplier
        self.ratio = ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df = df.reset_index(drop=True)

        df[self.column_name] = 0
        atr_transformer = ATR('ATR_tmp', self.ttl)
        df = atr_transformer.transform(df)
        
        for idx, row in df.iterrows():
            tp_long = row['close'] + self.multiplier * row['ATR_tmp']
            sl_long = row['close'] - (self.multiplier * row['ATR_tmp']) / self.ratio
            long_status = 'checking'

            tp_short = row['close'] - self.multiplier * row['ATR_tmp']
            sl_short = row['close'] + (self.multiplier * row['ATR_tmp']) / self.ratio
            short_status = 'checking'

            for i in range(1, self.ttl + 1):
                if idx + i >= len(df):
                    break

                if long_status != 'win' and short_status != 'win':
                    nxt_high = df.iloc[idx + i]['high']
                    nxt_low = df.iloc[idx + i]['low']

                    if long_status == 'checking':
                        if nxt_low <= sl_long:
                            long_status = 'lose'
                        elif nxt_high >= tp_long:
                            long_status = 'win'

                    if short_status == 'checking' and long_status != 'win':
                        if nxt_high >= sl_short:
                            short_status = 'lose'
                        elif nxt_low <= tp_short:
                            short_status = 'win'

            if long_status == 'win':
                df.loc[idx, self.column_name] = 1
            if short_status == 'win':
                df.loc[idx, self.column_name] = -1

        return X.assign(**{self.column_name: df[self.column_name]})


class OptimalTrade(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, rankings):
        self.column_name = column_name
        self.rankings = rankings

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df[self.column_name] = -1

        for idx, col in enumerate(self.rankings):
            condition = (df[col] != 0) & (df[self.column_name] == 0)
            df.loc[condition, self.column_name] = idx

        return X.assign(**{self.column_name: df[self.column_name]})



class BinaryTrendDirection(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, periods, shift=False):
        self.column_name = column_name
        self.periods = periods
        self.shift = shift

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        df[self.column_name] = 0

        trend_scorer = TrendDirectionPowerScorer('TREND_TMP', self.periods)
        df = trend_scorer.transform(df)

        df.loc[(df['TREND_TMP'] < 0), self.column_name] = -1
        df.loc[(df['TREND_TMP'] > 0), self.column_name] = 1

        if self.shift is True:
            df[self.column_name] = df[self.column_name].shift(-self.periods)

        return X.assign(**{self.column_name: df[self.column_name]})



class UpOrDown(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, periods):
        self.column_name = column_name
        self.periods = periods


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        df['DIFF_TMP'] = df['close'].shift(-self.periods) - df['close']
        df['DIFF_ZSCORE'] = (df['DIFF_TMP'] - df['DIFF_TMP'].mean()) / df['DIFF_TMP'].std()

        df[self.column_name] = np.tanh(df['DIFF_ZSCORE'])

        return X.assign(**{self.column_name: df[self.column_name]})