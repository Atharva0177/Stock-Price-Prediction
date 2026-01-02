"""
Portfolio Manager Module
Track virtual portfolio with buy/sell transactions and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manage virtual stock portfolio"""
    
    def __init__(self, initial_cash=100000):
        """
        Initialize portfolio
        
        Args:
            initial_cash (float): Initial cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}  # {ticker: {'shares': n, 'avg_price': price}}
        self.transactions = []
        self.portfolio_history = []
    
    def buy_stock(self, ticker, shares, price, date=None):
        """
        Buy stock shares
        
        Args:
            ticker (str): Stock ticker
            shares (int): Number of shares
            price (float): Price per share
            date (datetime): Transaction date
            
        Returns:
            dict: Transaction result
        """
        if date is None:
            date = datetime.now()
        
        total_cost = shares * price
        commission = total_cost * 0.001  # 0.1% commission
        total_with_commission = total_cost + commission
        
        if total_with_commission > self.cash:
            return {
                'success': False,
                'message': f'Insufficient funds. Need ${total_with_commission:.2f}, have ${self.cash:.2f}'
            }
        
        # Update cash
        self.cash -= total_with_commission
        
        # Update holdings
        if ticker in self.holdings:
            current_shares = self.holdings[ticker]['shares']
            current_value = current_shares * self.holdings[ticker]['avg_price']
            new_value = shares * price
            total_shares = current_shares + shares
            new_avg_price = (current_value + new_value) / total_shares
            
            self.holdings[ticker]['shares'] = total_shares
            self.holdings[ticker]['avg_price'] = new_avg_price
        else:
            self.holdings[ticker] = {
                'shares': shares,
                'avg_price': price
            }
        
        # Record transaction
        transaction = {
            'type': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'commission': commission,
            'total': total_with_commission,
            'date': date,
            'cash_after': self.cash
        }
        self.transactions.append(transaction)
        
        return {
            'success': True,
            'message': f'Bought {shares} shares of {ticker} at ${price:.2f}',
            'transaction': transaction
        }
    
    def sell_stock(self, ticker, shares, price, date=None):
        """
        Sell stock shares
        
        Args:
            ticker (str): Stock ticker
            shares (int): Number of shares
            price (float): Price per share
            date (datetime): Transaction date
            
        Returns:
            dict: Transaction result
        """
        if date is None:
            date = datetime.now()
        
        if ticker not in self.holdings:
            return {
                'success': False,
                'message': f'You do not own any shares of {ticker}'
            }
        
        if shares > self.holdings[ticker]['shares']:
            return {
                'success': False,
                'message': f'Cannot sell {shares} shares. You only own {self.holdings[ticker]["shares"]}'
            }
        
        total_proceeds = shares * price
        commission = total_proceeds * 0.001  # 0.1% commission
        net_proceeds = total_proceeds - commission
        
        # Calculate profit/loss
        cost_basis = shares * self.holdings[ticker]['avg_price']
        profit_loss = net_proceeds - cost_basis
        profit_loss_pct = (profit_loss / cost_basis) * 100
        
        # Update cash
        self.cash += net_proceeds
        
        # Update holdings
        self.holdings[ticker]['shares'] -= shares
        if self.holdings[ticker]['shares'] == 0:
            del self.holdings[ticker]
        
        # Record transaction
        transaction = {
            'type': 'SELL',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'commission': commission,
            'total': net_proceeds,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'date': date,
            'cash_after': self.cash
        }
        self.transactions.append(transaction)
        
        return {
            'success': True,
            'message': f'Sold {shares} shares of {ticker} at ${price:.2f}. P/L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)',
            'transaction': transaction
        }
    
    def get_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value
        
        Args:
            current_prices (dict): {ticker: current_price}
            
        Returns:
            dict: Portfolio value breakdown
        """
        holdings_value = 0
        holdings_detail = {}
        
        for ticker, holding in self.holdings.items():
            current_price = current_prices.get(ticker, holding['avg_price'])
            shares = holding['shares']
            value = shares * current_price
            cost_basis = shares * holding['avg_price']
            profit_loss = value - cost_basis
            profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            holdings_value += value
            holdings_detail[ticker] = {
                'shares': shares,
                'avg_price': holding['avg_price'],
                'current_price': current_price,
                'value': value,
                'cost_basis': cost_basis,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct
            }
        
        total_value = self.cash + holdings_value
        total_return = total_value - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        return {
            'cash': self.cash,
            'holdings_value': holdings_value,
            'total_value': total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'holdings_detail': holdings_detail
        }
    
    def get_transactions_df(self):
        """
        Get transactions as DataFrame
        
        Returns:
            pd.DataFrame: Transaction history
        """
        if not self.transactions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.transactions)
    
    def get_holdings_df(self, current_prices):
        """
        Get current holdings as DataFrame
        
        Args:
            current_prices (dict): {ticker: current_price}
            
        Returns:
            pd.DataFrame: Current holdings
        """
        if not self.holdings:
            return pd.DataFrame()
        
        portfolio_value = self.get_portfolio_value(current_prices)
        holdings_data = []
        
        for ticker, details in portfolio_value['holdings_detail'].items():
            holdings_data.append({
                'Ticker': ticker,
                'Shares': details['shares'],
                'Avg Price': details['avg_price'],
                'Current Price': details['current_price'],
                'Value': details['value'],
                'Cost Basis': details['cost_basis'],
                'P/L $': details['profit_loss'],
                'P/L %': details['profit_loss_pct']
            })
        
        return pd.DataFrame(holdings_data)
    
    def get_performance_metrics(self, current_prices):
        """
        Calculate performance metrics
        
        Args:
            current_prices (dict): {ticker: current_price}
            
        Returns:
            dict: Performance metrics
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        
        # Calculate metrics
        total_invested = self.initial_cash - self.cash + sum(
            holding['shares'] * holding['avg_price'] 
            for holding in self.holdings.values()
        )
        
        # Win/Loss ratio
        transactions_df = self.get_transactions_df()
        if not transactions_df.empty:
            sell_transactions = transactions_df[transactions_df['type'] == 'SELL']
            if not sell_transactions.empty:
                wins = len(sell_transactions[sell_transactions['profit_loss'] > 0])
                losses = len(sell_transactions[sell_transactions['profit_loss'] <= 0])
                win_rate = (wins / len(sell_transactions)) * 100 if len(sell_transactions) > 0 else 0
                avg_win = sell_transactions[sell_transactions['profit_loss'] > 0]['profit_loss'].mean() if wins > 0 else 0
                avg_loss = sell_transactions[sell_transactions['profit_loss'] <= 0]['profit_loss'].mean() if losses > 0 else 0
            else:
                wins = losses = win_rate = avg_win = avg_loss = 0
        else:
            wins = losses = win_rate = avg_win = avg_loss = 0
        
        return {
            'total_value': portfolio_value['total_value'],
            'total_return': portfolio_value['total_return'],
            'total_return_pct': portfolio_value['total_return_pct'],
            'cash': portfolio_value['cash'],
            'invested': total_invested,
            'num_holdings': len(self.holdings),
            'num_transactions': len(self.transactions),
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def to_dict(self):
        """Export portfolio to dictionary"""
        return {
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'holdings': self.holdings,
            'transactions': [
                {**t, 'date': t['date'].isoformat() if isinstance(t['date'], datetime) else t['date']}
                for t in self.transactions
            ]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Import portfolio from dictionary"""
        portfolio = cls(initial_cash=data['initial_cash'])
        portfolio.cash = data['cash']
        portfolio.holdings = data['holdings']
        portfolio.transactions = [
            {**t, 'date': datetime.fromisoformat(t['date']) if isinstance(t['date'], str) else t['date']}
            for t in data['transactions']
        ]
        return portfolio
    
    def save_to_json(self, filepath):
        """Save portfolio to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath):
        """Load portfolio from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
