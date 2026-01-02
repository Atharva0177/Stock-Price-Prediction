"""
Risk Metrics Module
Calculate comprehensive risk metrics for stocks and portfolios
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Calculate risk metrics for stocks"""
    
    @staticmethod
    def calculate_returns(prices):
        """Calculate percentage returns"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_volatility(returns, periods_per_year=252):
        """
        Calculate annualized volatility
        
        Args:
            returns (pd.Series): Returns series
            periods_per_year (int): Trading days per year
            
        Returns:
            float: Annualized volatility
        """
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns (pd.Series): Returns series
            confidence (float): Confidence level (0.95 or 0.99)
            
        Returns:
            float: VaR value
        """
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns, confidence=0.95):
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
        
        Args:
            returns (pd.Series): Returns series
            confidence (float): Confidence level
            
        Returns:
            float: CVaR value
        """
        var = RiskAnalyzer.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(prices):
        """
        Calculate maximum drawdown
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            dict: Max drawdown info
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_date = running_max[:max_dd_date].idxmax()
        
        # Find recovery date (if any)
        recovery_date = None
        if max_dd_date < drawdown.index[-1]:
            after_dd = drawdown[max_dd_date:]
            recovered = after_dd[after_dd >= 0]
            if not recovered.empty:
                recovery_date = recovered.index[0]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'peak_date': str(peak_date.date()) if hasattr(peak_date, 'date') else str(peak_date),
            'trough_date': str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
            'recovery_date': str(recovery_date.date()) if recovery_date and hasattr(recovery_date, 'date') else (str(recovery_date) if recovery_date else None),
            'drawdown_series': drawdown
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate Sharpe Ratio
        
        Args:
            returns (pd.Series): Returns series
            risk_free_rate (float): Annual risk-free rate
            periods_per_year (int): Trading periods per year
            
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
        """
        Calculate Sortino Ratio (uses downside deviation)
        
        Args:
            returns (pd.Series): Returns series
            risk_free_rate (float): Annual risk-free rate
            periods_per_year (int): Trading periods per year
            
        Returns:
            float: Sortino ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_calmar_ratio(returns, prices):
        """
        Calculate Calmar Ratio (return / max drawdown)
        
        Args:
            returns (pd.Series): Returns series
            prices (pd.Series): Price series
            
        Returns:
            float: Calmar ratio
        """
        annual_return = returns.mean() * 252
        max_dd = RiskAnalyzer.calculate_max_drawdown(prices)['max_drawdown']
        
        if max_dd == 0:
            return 0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_beta(stock_returns, market_returns):
        """
        Calculate Beta (systematic risk)
        
        Args:
            stock_returns (pd.Series): Stock returns
            market_returns (pd.Series): Market returns
            
        Returns:
            float: Beta value
        """
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_skewness(returns):
        """Calculate skewness of returns"""
        return stats.skew(returns)
    
    @staticmethod
    def calculate_kurtosis(returns):
        """Calculate kurtosis of returns"""
        return stats.kurtosis(returns)
    
    @classmethod
    def get_comprehensive_risk_metrics(cls, df, market_df=None, risk_free_rate=0.02):
        """
        Calculate all risk metrics
        
        Args:
            df (pd.DataFrame): Stock data with Close prices
            market_df (pd.DataFrame): Market data for beta calculation
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            dict: Comprehensive risk metrics
        """
        prices = df['Close']
        returns = cls.calculate_returns(prices)
        
        # Basic metrics
        volatility = cls.calculate_volatility(returns)
        var_95 = cls.calculate_var(returns, 0.95)
        var_99 = cls.calculate_var(returns, 0.99)
        cvar_95 = cls.calculate_cvar(returns, 0.95)
        cvar_99 = cls.calculate_cvar(returns, 0.99)
        
        # Drawdown
        drawdown_info = cls.calculate_max_drawdown(prices)
        
        # Risk-adjusted returns
        sharpe = cls.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = cls.calculate_sortino_ratio(returns, risk_free_rate)
        calmar = cls.calculate_calmar_ratio(returns, prices)
        
        # Distribution metrics
        skewness = cls.calculate_skewness(returns)
        kurtosis = cls.calculate_kurtosis(returns)
        
        # Beta (if market data available)
        beta = None
        if market_df is not None and 'Close' in market_df.columns:
            market_returns = cls.calculate_returns(market_df['Close'])
            # Align dates
            aligned = pd.concat([returns, market_returns], axis=1, join='inner')
            if len(aligned) > 0:
                beta = cls.calculate_beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
        
        return {
            'volatility': volatility,
            'annualized_volatility_pct': volatility * 100,
            'var_95': var_95,
            'var_95_pct': var_95 * 100,
            'var_99': var_99,
            'var_99_pct': var_99 * 100,
            'cvar_95': cvar_95,
            'cvar_95_pct': cvar_95 * 100,
            'cvar_99': cvar_99,
            'cvar_99_pct': cvar_99 * 100,
            'max_drawdown': drawdown_info['max_drawdown'],
            'max_drawdown_pct': drawdown_info['max_drawdown_pct'],
            'peak_date': drawdown_info['peak_date'],
            'trough_date': drawdown_info['trough_date'],
            'recovery_date': drawdown_info['recovery_date'],
            'drawdown_series': drawdown_info['drawdown_series'],
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': beta,
            'mean_return': returns.mean() * 252,
            'mean_return_pct': returns.mean() * 252 * 100
        }
    
    @staticmethod
    def get_risk_rating(metrics):
        """
        Get overall risk rating based on metrics
        
        Args:
            metrics (dict): Risk metrics
            
        Returns:
            dict: Risk rating info
        """
        score = 0
        total = 0
        
        # Volatility (lower is better)
        vol = metrics['annualized_volatility_pct']
        if vol < 15:
            score += 5
        elif vol < 25:
            score += 4
        elif vol < 35:
            score += 3
        elif vol < 50:
            score += 2
        else:
            score += 1
        total += 5
        
        # Sharpe Ratio (higher is better)
        sharpe = metrics['sharpe_ratio']
        if sharpe > 2:
            score += 5
        elif sharpe > 1:
            score += 4
        elif sharpe > 0.5:
            score += 3
        elif sharpe > 0:
            score += 2
        else:
            score += 1
        total += 5
        
        # Max Drawdown (smaller is better)
        dd = abs(metrics['max_drawdown_pct'])
        if dd < 10:
            score += 5
        elif dd < 20:
            score += 4
        elif dd < 30:
            score += 3
        elif dd < 50:
            score += 2
        else:
            score += 1
        total += 5
        
        # Calculate percentage
        percentage = (score / total) * 100
        
        # Determine rating
        if percentage >= 80:
            rating = 'Low Risk'
            color = '#10b981'
        elif percentage >= 60:
            rating = 'Moderate Risk'
            color = '#f59e0b'
        elif percentage >= 40:
            rating = 'High Risk'
            color = '#ef4444'
        else:
            rating = 'Very High Risk'
            color = '#dc2626'
        
        return {
            'rating': rating,
            'score': score,
            'total': total,
            'percentage': percentage,
            'color': color
        }
