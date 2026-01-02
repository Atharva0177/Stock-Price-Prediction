"""
Feature Engineering Module
Add technical indicators to stock data
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Add technical indicators and features to stock data"""
    
    @staticmethod
    def add_moving_averages(df, windows=[50, 200]):
        """
        Add moving averages to dataframe
        
        Args: 
            df (pd.DataFrame): Stock data
            windows (list): List of window sizes for moving averages
            
        Returns: 
            pd.DataFrame: Data with moving averages added
        """
        df = df.copy()
        for window in windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            logger.info(f"Added MA_{window}")
        return df
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df (pd.DataFrame): Stock data
            period (int): RSI period
            
        Returns: 
            pd.DataFrame: Data with RSI added
        """
        df = df.copy()
        
        # Calculate price changes
        delta = df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        logger.info(f"Added RSI_{period}")
        return df
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args: 
            df (pd.DataFrame): Stock data
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
            
        Returns: 
            pd.DataFrame: Data with MACD indicators added
        """
        df = df.copy()
        
        # Calculate EMAs
        ema_fast = df['Close']. ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close']. ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        df['MACD'] = ema_fast - ema_slow
        
        # Calculate signal line
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # Calculate MACD histogram
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        logger.info("Added MACD indicators")
        return df
    
    @staticmethod
    def calculate_bollinger_bands(df, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Args:
            df (pd. DataFrame): Stock data
            window (int): Moving average window
            num_std (int): Number of standard deviations
            
        Returns:
            pd.DataFrame: Data with Bollinger Bands added
        """
        df = df.copy()
        
        # Calculate middle band (SMA)
        df['BB_Middle'] = df['Close']. rolling(window=window).mean()
        
        # Calculate standard deviation
        std = df['Close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        df['BB_Upper'] = df['BB_Middle'] + (std * num_std)
        df['BB_Lower'] = df['BB_Middle'] - (std * num_std)
        
        logger.info("Added Bollinger Bands")
        return df
    
    @staticmethod
    def add_volume_features(df):
        """
        Add volume-based features
        
        Args: 
            df (pd.DataFrame): Stock data
            
        Returns: 
            pd.DataFrame: Data with volume features added
        """
        df = df.copy()
        
        # Volume moving average
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Volume ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        logger.info("Added volume features")
        return df
    
    @staticmethod
    def add_price_features(df):
        """
        Add price-based features
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with price features added
        """
        df = df.copy()
        
        # Daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Price change
        df['Price_Change'] = df['Close'].diff()
        
        # High-Low range
        df['HL_Range'] = df['High'] - df['Low']
        
        # Close-Open range
        df['CO_Range'] = df['Close'] - df['Open']
        
        logger.info("Added price features")
        return df
    
    @classmethod
    def add_all_indicators(cls, df, ma_windows=[50, 200], rsi_period=14):
        """
        Add all technical indicators
        
        Args:
            df (pd.DataFrame): Stock data
            ma_windows (list): Moving average windows
            rsi_period (int): RSI period
            
        Returns:
            pd. DataFrame: Data with all indicators added
        """
        df = cls.add_moving_averages(df, windows=ma_windows)
        df = cls.calculate_rsi(df, period=rsi_period)
        df = cls.calculate_macd(df)
        df = cls.calculate_bollinger_bands(df)
        df = cls.add_volume_features(df)
        df = cls.add_price_features(df)
        
        # Drop NaN values created by rolling windows
        df. dropna(inplace=True)
        
        logger.info("All indicators added successfully")
        return df