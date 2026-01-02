"""
Trading Signals Module
Generate buy/sell recommendations based on technical indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingSignals:
    """Generate trading signals from technical indicators"""
    
    @staticmethod
    def generate_rsi_signal(rsi_value, oversold=30, overbought=70):
        """
        Generate signal based on RSI
        
        Args:
            rsi_value (float): Current RSI value
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
            
        Returns:
            dict: Signal information
        """
        if rsi_value < oversold:
            return {
                'signal': 'BUY',
                'strength': 'STRONG',
                'reason': f'RSI is oversold at {rsi_value:.2f}',
                'indicator': 'RSI'
            }
        elif rsi_value < 40:
            return {
                'signal': 'BUY',
                'strength': 'MODERATE',
                'reason': f'RSI indicates potential upside at {rsi_value:.2f}',
                'indicator': 'RSI'
            }
        elif rsi_value > overbought:
            return {
                'signal': 'SELL',
                'strength': 'STRONG',
                'reason': f'RSI is overbought at {rsi_value:.2f}',
                'indicator': 'RSI'
            }
        elif rsi_value > 60:
            return {
                'signal': 'SELL',
                'strength': 'MODERATE',
                'reason': f'RSI indicates potential downside at {rsi_value:.2f}',
                'indicator': 'RSI'
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 'NEUTRAL',
                'reason': f'RSI is neutral at {rsi_value:.2f}',
                'indicator': 'RSI'
            }
    
    @staticmethod
    def generate_macd_signal(macd, signal_line, histogram):
        """
        Generate signal based on MACD
        
        Args:
            macd (float): MACD value
            signal_line (float): Signal line value
            histogram (float): MACD histogram
            
        Returns:
            dict: Signal information
        """
        if macd > signal_line and histogram > 0:
            strength = 'STRONG' if histogram > 0.5 else 'MODERATE'
            return {
                'signal': 'BUY',
                'strength': strength,
                'reason': f'MACD bullish crossover (Histogram: {histogram:.3f})',
                'indicator': 'MACD'
            }
        elif macd < signal_line and histogram < 0:
            strength = 'STRONG' if histogram < -0.5 else 'MODERATE'
            return {
                'signal': 'SELL',
                'strength': strength,
                'reason': f'MACD bearish crossover (Histogram: {histogram:.3f})',
                'indicator': 'MACD'
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 'NEUTRAL',
                'reason': f'MACD shows no clear trend (Histogram: {histogram:.3f})',
                'indicator': 'MACD'
            }
    
    @staticmethod
    def generate_ma_signal(current_price, ma_50, ma_200):
        """
        Generate signal based on Moving Average crossover
        
        Args:
            current_price (float): Current stock price
            ma_50 (float): 50-day moving average
            ma_200 (float): 200-day moving average
            
        Returns:
            dict: Signal information
        """
        if ma_50 > ma_200 and current_price > ma_50:
            return {
                'signal': 'BUY',
                'strength': 'STRONG',
                'reason': 'Golden Cross: Price above both MAs with bullish trend',
                'indicator': 'Moving Averages'
            }
        elif ma_50 > ma_200 and current_price < ma_50:
            return {
                'signal': 'HOLD',
                'strength': 'MODERATE',
                'reason': 'Bullish MA alignment but price below MA50',
                'indicator': 'Moving Averages'
            }
        elif ma_50 < ma_200 and current_price < ma_50:
            return {
                'signal': 'SELL',
                'strength': 'STRONG',
                'reason': 'Death Cross: Price below both MAs with bearish trend',
                'indicator': 'Moving Averages'
            }
        elif ma_50 < ma_200 and current_price > ma_50:
            return {
                'signal': 'HOLD',
                'strength': 'MODERATE',
                'reason': 'Bearish MA alignment but price above MA50',
                'indicator': 'Moving Averages'
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 'NEUTRAL',
                'reason': 'Mixed moving average signals',
                'indicator': 'Moving Averages'
            }
    
    @staticmethod
    def generate_bollinger_signal(current_price, bb_upper, bb_lower, bb_middle):
        """
        Generate signal based on Bollinger Bands
        
        Args:
            current_price (float): Current stock price
            bb_upper (float): Upper Bollinger Band
            bb_lower (float): Lower Bollinger Band
            bb_middle (float): Middle Bollinger Band
            
        Returns:
            dict: Signal information
        """
        if current_price <= bb_lower:
            return {
                'signal': 'BUY',
                'strength': 'STRONG',
                'reason': f'Price at lower Bollinger Band ({current_price:.2f})',
                'indicator': 'Bollinger Bands'
            }
        elif current_price < bb_middle and current_price > bb_lower:
            return {
                'signal': 'BUY',
                'strength': 'MODERATE',
                'reason': f'Price below middle band, potential bounce',
                'indicator': 'Bollinger Bands'
            }
        elif current_price >= bb_upper:
            return {
                'signal': 'SELL',
                'strength': 'STRONG',
                'reason': f'Price at upper Bollinger Band ({current_price:.2f})',
                'indicator': 'Bollinger Bands'
            }
        elif current_price > bb_middle and current_price < bb_upper:
            return {
                'signal': 'SELL',
                'strength': 'MODERATE',
                'reason': f'Price above middle band, potential pullback',
                'indicator': 'Bollinger Bands'
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 'NEUTRAL',
                'reason': f'Price near middle band',
                'indicator': 'Bollinger Bands'
            }
    
    @staticmethod
    def generate_volume_signal(current_volume, avg_volume):
        """
        Generate signal based on volume analysis
        
        Args:
            current_volume (float): Current volume
            avg_volume (float): Average volume
            
        Returns:
            dict: Signal information
        """
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.5:
            return {
                'signal': 'STRONG_TREND',
                'strength': 'STRONG',
                'reason': f'High volume confirms trend ({volume_ratio:.2f}x average)',
                'indicator': 'Volume'
            }
        elif volume_ratio > 1.2:
            return {
                'signal': 'MODERATE_TREND',
                'strength': 'MODERATE',
                'reason': f'Above average volume ({volume_ratio:.2f}x)',
                'indicator': 'Volume'
            }
        elif volume_ratio < 0.7:
            return {
                'signal': 'WEAK_TREND',
                'strength': 'WEAK',
                'reason': f'Low volume, weak conviction ({volume_ratio:.2f}x)',
                'indicator': 'Volume'
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'strength': 'NEUTRAL',
                'reason': f'Normal volume ({volume_ratio:.2f}x average)',
                'indicator': 'Volume'
            }
    
    @classmethod
    def get_comprehensive_signals(cls, df):
        """
        Generate comprehensive trading signals from all indicators
        
        Args:
            df (pd.DataFrame): DataFrame with technical indicators
            
        Returns:
            dict: All signals and overall recommendation
        """
        # Get latest values
        latest = df.iloc[-1]
        
        # Individual signals
        signals = {}
        
        # RSI Signal
        if 'RSI' in df.columns:
            signals['RSI'] = cls.generate_rsi_signal(latest['RSI'])
        
        # MACD Signal
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            macd_hist = latest['MACD'] - latest['MACD_Signal']
            signals['MACD'] = cls.generate_macd_signal(
                latest['MACD'], 
                latest['MACD_Signal'],
                macd_hist
            )
        
        # Moving Average Signal
        if all(col in df.columns for col in ['Close', 'MA_50', 'MA_200']):
            signals['MA'] = cls.generate_ma_signal(
                latest['Close'],
                latest['MA_50'],
                latest['MA_200']
            )
        
        # Bollinger Bands Signal
        if all(col in df.columns for col in ['Close', 'BB_Upper', 'BB_Lower', 'MA_20']):
            bb_middle = latest.get('MA_20', (latest['BB_Upper'] + latest['BB_Lower']) / 2)
            signals['BB'] = cls.generate_bollinger_signal(
                latest['Close'],
                latest['BB_Upper'],
                latest['BB_Lower'],
                bb_middle
            )
        
        # Volume Signal
        if 'Volume_Ratio' in df.columns:
            avg_volume = df['Volume'].mean()
            signals['Volume'] = cls.generate_volume_signal(
                latest['Volume'],
                avg_volume
            )
        
        # Calculate overall recommendation
        overall = cls._calculate_overall_signal(signals)
        
        return {
            'signals': signals,
            'overall': overall,
            'timestamp': datetime.now(),
            'current_price': latest['Close']
        }
    
    @staticmethod
    def _calculate_overall_signal(signals):
        """
        Calculate overall signal from individual indicators
        
        Args:
            signals (dict): Individual indicator signals
            
        Returns:
            dict: Overall recommendation
        """
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        weights = {
            'RSI': 1.5,
            'MACD': 2.0,
            'MA': 2.5,
            'BB': 1.5,
            'Volume': 1.0
        }
        
        for indicator, signal in signals.items():
            if indicator == 'Volume':
                continue  # Volume is confirmatory, not directional
                
            weight = weights.get(indicator, 1.0)
            total_weight += weight
            
            if signal['signal'] == 'BUY':
                strength_multiplier = 1.5 if signal['strength'] == 'STRONG' else 1.0
                buy_score += weight * strength_multiplier
            elif signal['signal'] == 'SELL':
                strength_multiplier = 1.5 if signal['strength'] == 'STRONG' else 1.0
                sell_score += weight * strength_multiplier
        
        # Normalize scores
        if total_weight > 0:
            buy_score = (buy_score / total_weight) * 100
            sell_score = (sell_score / total_weight) * 100
        
        # Determine overall signal
        diff = buy_score - sell_score
        
        if diff > 30:
            recommendation = 'STRONG BUY'
            confidence = min(95, 60 + abs(diff)/2)
        elif diff > 15:
            recommendation = 'BUY'
            confidence = min(85, 50 + abs(diff)/2)
        elif diff < -30:
            recommendation = 'STRONG SELL'
            confidence = min(95, 60 + abs(diff)/2)
        elif diff < -15:
            recommendation = 'SELL'
            confidence = min(85, 50 + abs(diff)/2)
        else:
            recommendation = 'HOLD'
            confidence = 50 + (20 - abs(diff))
        
        return {
            'recommendation': recommendation,
            'confidence': round(confidence, 1),
            'buy_score': round(buy_score, 1),
            'sell_score': round(sell_score, 1),
            'summary': f'{recommendation} with {confidence:.0f}% confidence'
        }
    
    @staticmethod
    def get_signal_color(signal):
        """Get color for signal display"""
        colors = {
            'STRONG BUY': '#00ff00',
            'BUY': '#90EE90',
            'HOLD': '#FFD700',
            'SELL': '#FFA500',
            'STRONG SELL': '#ff0000'
        }
        return colors.get(signal, '#808080')
    
    @staticmethod
    def get_signal_emoji(signal):
        """Get emoji for signal display"""
        emojis = {
            'STRONG BUY': '🚀',
            'BUY': '📈',
            'HOLD': '⏸️',
            'SELL': '📉',
            'STRONG SELL': '⚠️'
        }
        return emojis.get(signal, '➖')
