"""
Pattern Recognition Module
Identify candlestick patterns and chart patterns in stock data
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternRecognizer:
    """Recognize technical patterns in stock data"""
    
    @staticmethod
    def detect_doji(df, threshold=0.1):
        """
        Detect Doji candlestick pattern
        
        Args:
            df (pd.DataFrame): OHLC data
            threshold (float): Body size threshold as % of range
            
        Returns:
            pd.Series: Boolean series indicating Doji patterns
        """
        body = abs(df['Close'] - df['Open'])
        range_size = df['High'] - df['Low']
        
        # Avoid division by zero
        range_size = range_size.replace(0, np.nan)
        
        return (body / range_size) < threshold
    
    @staticmethod
    def detect_hammer(df, body_ratio=0.3, shadow_ratio=2):
        """
        Detect Hammer pattern (bullish reversal)
        
        Args:
            df (pd.DataFrame): OHLC data
            body_ratio (float): Max body size relative to range
            shadow_ratio (float): Min lower shadow to body ratio
            
        Returns:
            pd.Series: Boolean series indicating Hammer patterns
        """
        body = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        
        # Avoid division by zero
        body = body.replace(0, 0.001)
        total_range = total_range.replace(0, np.nan)
        
        conditions = (
            (body / total_range < body_ratio) &
            (lower_shadow / body > shadow_ratio) &
            (upper_shadow < body)
        )
        
        return conditions
    
    @staticmethod
    def detect_shooting_star(df, body_ratio=0.3, shadow_ratio=2):
        """
        Detect Shooting Star pattern (bearish reversal)
        
        Args:
            df (pd.DataFrame): OHLC data
            body_ratio (float): Max body size relative to range
            shadow_ratio (float): Min upper shadow to body ratio
            
        Returns:
            pd.Series: Boolean series indicating Shooting Star patterns
        """
        body = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Avoid division by zero
        body = body.replace(0, 0.001)
        total_range = total_range.replace(0, np.nan)
        
        conditions = (
            (body / total_range < body_ratio) &
            (upper_shadow / body > shadow_ratio) &
            (lower_shadow < body)
        )
        
        return conditions
    
    @staticmethod
    def detect_engulfing(df):
        """
        Detect Bullish and Bearish Engulfing patterns
        
        Returns:
            tuple: (bullish_engulfing, bearish_engulfing) Series
        """
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)
        
        # Bullish Engulfing
        bullish = (
            (prev_close < prev_open) &  # Previous candle was bearish
            (df['Close'] > df['Open']) &  # Current candle is bullish
            (df['Open'] < prev_close) &  # Opens below previous close
            (df['Close'] > prev_open)  # Closes above previous open
        )
        
        # Bearish Engulfing
        bearish = (
            (prev_close > prev_open) &  # Previous candle was bullish
            (df['Close'] < df['Open']) &  # Current candle is bearish
            (df['Open'] > prev_close) &  # Opens above previous close
            (df['Close'] < prev_open)  # Closes below previous open
        )
        
        return bullish, bearish
    
    @staticmethod
    def detect_morning_star(df):
        """
        Detect Morning Star pattern (bullish reversal, 3-candle pattern)
        
        Returns:
            pd.Series: Boolean series indicating Morning Star patterns
        """
        # First candle: Large bearish
        first_bearish = df['Close'].shift(2) < df['Open'].shift(2)
        first_large = abs(df['Close'].shift(2) - df['Open'].shift(2)) > \
                     (df['High'].shift(2) - df['Low'].shift(2)) * 0.6
        
        # Second candle: Small body (star)
        second_small = abs(df['Close'].shift(1) - df['Open'].shift(1)) < \
                      (df['High'].shift(1) - df['Low'].shift(1)) * 0.3
        second_gaps = df['High'].shift(1) < df['Close'].shift(2)
        
        # Third candle: Large bullish
        third_bullish = df['Close'] > df['Open']
        third_large = abs(df['Close'] - df['Open']) > (df['High'] - df['Low']) * 0.6
        third_closes_high = df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2
        
        return first_bearish & first_large & second_small & second_gaps & \
               third_bullish & third_large & third_closes_high
    
    @staticmethod
    def detect_evening_star(df):
        """
        Detect Evening Star pattern (bearish reversal, 3-candle pattern)
        
        Returns:
            pd.Series: Boolean series indicating Evening Star patterns
        """
        # First candle: Large bullish
        first_bullish = df['Close'].shift(2) > df['Open'].shift(2)
        first_large = abs(df['Close'].shift(2) - df['Open'].shift(2)) > \
                     (df['High'].shift(2) - df['Low'].shift(2)) * 0.6
        
        # Second candle: Small body (star)
        second_small = abs(df['Close'].shift(1) - df['Open'].shift(1)) < \
                      (df['High'].shift(1) - df['Low'].shift(1)) * 0.3
        second_gaps = df['Low'].shift(1) > df['Close'].shift(2)
        
        # Third candle: Large bearish
        third_bearish = df['Close'] < df['Open']
        third_large = abs(df['Close'] - df['Open']) > (df['High'] - df['Low']) * 0.6
        third_closes_low = df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2
        
        return first_bullish & first_large & second_small & second_gaps & \
               third_bearish & third_large & third_closes_low
    
    @staticmethod
    def detect_support_resistance(df, window=20, tolerance=0.02):
        """
        Identify support and resistance levels
        
        Args:
            df (pd.DataFrame): Price data
            window (int): Window for local min/max
            tolerance (float): Price tolerance for level grouping
            
        Returns:
            dict: Support and resistance levels
        """
        prices = df['Close']
        
        # Find local minima (support)
        local_min = prices[(prices.shift(1) > prices) & (prices.shift(-1) > prices)]
        
        # Find local maxima (resistance)
        local_max = prices[(prices.shift(1) < prices) & (prices.shift(-1) < prices)]
        
        # Group similar levels
        def group_levels(levels, tolerance):
            if len(levels) == 0:
                return []
            
            sorted_levels = sorted(levels)
            grouped = [[sorted_levels[0]]]
            
            for level in sorted_levels[1:]:
                if abs(level - grouped[-1][-1]) / grouped[-1][-1] < tolerance:
                    grouped[-1].append(level)
                else:
                    grouped.append([level])
            
            return [np.mean(group) for group in grouped]
        
        support_levels = group_levels(local_min.tolist(), tolerance)
        resistance_levels = group_levels(local_max.tolist(), tolerance)
        
        # Get most recent levels
        current_price = prices.iloc[-1]
        
        # Find nearest support (below current price)
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        
        # Find nearest resistance (above current price)
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        
        return {
            'support_levels': sorted(support_levels),
            'resistance_levels': sorted(resistance_levels),
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': current_price
        }
    
    @classmethod
    def detect_all_patterns(cls, df):
        """
        Detect all candlestick patterns
        
        Args:
            df (pd.DataFrame): OHLC data
            
        Returns:
            dict: All detected patterns
        """
        patterns = {}
        
        # Single candle patterns
        patterns['Doji'] = cls.detect_doji(df)
        patterns['Hammer'] = cls.detect_hammer(df)
        patterns['Shooting Star'] = cls.detect_shooting_star(df)
        
        # Multi-candle patterns
        bullish_eng, bearish_eng = cls.detect_engulfing(df)
        patterns['Bullish Engulfing'] = bullish_eng
        patterns['Bearish Engulfing'] = bearish_eng
        patterns['Morning Star'] = cls.detect_morning_star(df)
        patterns['Evening Star'] = cls.detect_evening_star(df)
        
        # Count occurrences in last 30 days
        recent_patterns = {}
        for name, series in patterns.items():
            recent_count = series.tail(30).sum()
            if recent_count > 0:
                recent_patterns[name] = {
                    'count': int(recent_count),
                    'last_occurrence': series[series].index[-1] if series.any() else None,
                    'bullish': name in ['Hammer', 'Bullish Engulfing', 'Morning Star'],
                    'bearish': name in ['Shooting Star', 'Bearish Engulfing', 'Evening Star'],
                    'neutral': name in ['Doji']
                }
        
        # Support/Resistance
        sr_levels = cls.detect_support_resistance(df)
        
        return {
            'patterns': recent_patterns,
            'support_resistance': sr_levels,
            'pattern_series': patterns
        }
    
    @staticmethod
    def get_pattern_signal(patterns_dict):
        """
        Get overall signal from patterns
        
        Args:
            patterns_dict (dict): Dictionary of detected patterns
            
        Returns:
            dict: Overall pattern signal
        """
        bullish_score = 0
        bearish_score = 0
        
        for pattern_name, pattern_info in patterns_dict['patterns'].items():
            weight = pattern_info['count']
            
            if pattern_info['bullish']:
                bullish_score += weight
            elif pattern_info['bearish']:
                bearish_score += weight
        
        total = bullish_score + bearish_score
        
        if total == 0:
            return {
                'signal': 'NEUTRAL',
                'strength': 0,
                'bullish_score': 0,
                'bearish_score': 0,
                'description': 'No significant patterns detected'
            }
        
        bullish_pct = (bullish_score / total) * 100
        bearish_pct = (bearish_score / total) * 100
        
        if bullish_pct > 65:
            signal = 'BULLISH'
            strength = bullish_pct
            description = 'Strong bullish patterns detected'
        elif bearish_pct > 65:
            signal = 'BEARISH'
            strength = bearish_pct
            description = 'Strong bearish patterns detected'
        elif bullish_pct > bearish_pct:
            signal = 'MILDLY BULLISH'
            strength = bullish_pct
            description = 'Some bullish patterns present'
        elif bearish_pct > bullish_pct:
            signal = 'MILDLY BEARISH'
            strength = bearish_pct
            description = 'Some bearish patterns present'
        else:
            signal = 'NEUTRAL'
            strength = 50
            description = 'Mixed pattern signals'
        
        return {
            'signal': signal,
            'strength': strength,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'bullish_pct': bullish_pct,
            'bearish_pct': bearish_pct,
            'description': description
        }
