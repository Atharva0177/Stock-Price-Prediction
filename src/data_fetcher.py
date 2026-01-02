"""
Data Fetcher Module
Handles fetching stock data from Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging. INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch and manage stock data from Yahoo Finance"""
    
    def __init__(self, cache_dir='data/'):
        """
        Initialize the data fetcher
        
        Args:
            cache_dir (str): Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_stock_data(self, ticker, period='5y', interval='1d', start_date=None, end_date=None):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd. DataFrame: Stock data with OHLCV columns
        """
        try: 
            logger.info(f"Fetching data for {ticker}...")
            
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} rows of data for {ticker}")
            return df
        
        except Exception as e: 
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
    
    def get_stock_info(self, ticker):
        """
        Get stock information and metadata
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio':  info.get('trailingPE', 'N/A')
            }
        except Exception as e:
            logger. error(f"Error fetching info for {ticker}: {str(e)}")
            return {'name': ticker}
    
    def save_to_csv(self, df, filename):
        """
        Save DataFrame to CSV file
        
        Args: 
            df (pd.DataFrame): Data to save
            filename (str): Filename to save to
        """
        filepath = os.path.join(self.cache_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def load_from_csv(self, filename):
        """
        Load DataFrame from CSV file
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            pd.DataFrame: Loaded data
        """
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            logger.info(f"Data loaded from {filepath}")
            return df
        else:
            raise FileNotFoundError(f"File {filepath} not found")
    
    def is_cached(self, ticker, period='5y'):
        """
        Check if data is already cached
        
        Args: 
            ticker (str): Stock ticker symbol
            period (str): Data period
            
        Returns:
            bool: True if cached, False otherwise
        """
        filename = f"{ticker}_{period}. csv"
        filepath = os. path.join(self.cache_dir, filename)
        return os.path.exists(filepath)    
    def get_realtime_quote(self, ticker):
        """
        Get real-time stock quote with current price and intraday data
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Real-time quote data including current price, change, volume, etc.
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get current quote info
            info = stock.info
            
            # Get latest intraday data (1 minute intervals for last 5 days)
            intraday = stock.history(period='5d', interval='1m')
            
            if not intraday.empty:
                latest = intraday.iloc[-1]
                previous_close = info.get('previousClose', latest['Close'])
                current_price = latest['Close']
                change = current_price - previous_close
                change_pct = (change / previous_close) * 100 if previous_close else 0
                
                return {
                    'ticker': ticker,
                    'current_price': float(current_price),
                    'change': float(change),
                    'change_percent': float(change_pct),
                    'volume': int(latest['Volume']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'open': float(latest['Open']),
                    'previous_close': float(previous_close),
                    'timestamp': intraday.index[-1],
                    'market_cap': info.get('marketCap', 'N/A'),
                    'avg_volume': info.get('averageVolume', 'N/A')
                }
            else:
                # Fallback to regular info if intraday data not available
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                previous_close = info.get('previousClose', current_price)
                change = current_price - previous_close
                change_pct = (change / previous_close) * 100 if previous_close else 0
                
                return {
                    'ticker': ticker,
                    'current_price': float(current_price),
                    'change': float(change),
                    'change_percent': float(change_pct),
                    'volume': info.get('volume', 0),
                    'high': info.get('dayHigh', 0),
                    'low': info.get('dayLow', 0),
                    'open': info.get('open', 0),
                    'previous_close': float(previous_close),
                    'timestamp': datetime.now(),
                    'market_cap': info.get('marketCap', 'N/A'),
                    'avg_volume': info.get('averageVolume', 'N/A')
                }
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {ticker}: {str(e)}")
            raise
    
    def get_intraday_data(self, ticker, interval='1m', period='1d'):
        """
        Get intraday data for real-time charting
        
        Args:
            ticker (str): Stock ticker symbol
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m')
            period (str): Period ('1d', '5d', '1mo')
            
        Returns:
            pd.DataFrame: Intraday stock data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No intraday data found for ticker {ticker}")
            
            df.reset_index(inplace=True)
            logger.info(f"Successfully fetched {len(df)} intraday data points for {ticker}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching intraday data for {ticker}: {str(e)}")
            raise