"""
Visualization Module
Create plots and charts for stock data and predictions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Create visualizations for stock analysis"""
    
    @staticmethod
    def plot_price_history(df, ticker, figsize=(14, 7)):
        """
        Plot historical stock prices
        
        Args: 
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker
            figsize (tuple): Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
        ax.set_title(f'{ticker} Stock Price History', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_history(history, figsize=(14, 5)):
        """
        Plot training and validation loss
        
        Args: 
            history: Training history object (supports both Keras and N-BEATS formats)
            figsize (tuple): Figure size
        """
        # Check if history has 'history' attribute (Keras) or is a dict (N-BEATS)
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history
        
        # Check if MAE metrics are available
        has_mae = 'mae' in hist_dict and 'val_mae' in hist_dict
        
        if has_mae:
            # Two subplots for loss and MAE
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Loss plot
            ax1.plot(hist_dict['loss'], label='Training Loss', linewidth=2)
            ax1.plot(hist_dict['val_loss'], label='Validation Loss', linewidth=2)
            ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # MAE plot
            ax2.plot(hist_dict['mae'], label='Training MAE', linewidth=2)
            ax2.plot(hist_dict['val_mae'], label='Validation MAE', linewidth=2)
            ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('MAE', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Single plot for loss only (N-BEATS format)
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]/2, figsize[1]))
            
            # Use 'train_loss' and 'val_loss' keys if available, otherwise 'loss' and 'val_loss'
            train_key = 'train_loss' if 'train_loss' in hist_dict else 'loss'
            val_key = 'val_loss'
            
            ax1.plot(hist_dict[train_key], label='Training Loss', linewidth=2)
            ax1.plot(hist_dict[val_key], label='Validation Loss', linewidth=2)
            ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss (MSE)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_predictions(y_test, y_pred, dates=None, figsize=(14, 7)):
        """
        Plot actual vs predicted prices
        
        Args:
            y_test (np.array): Actual values
            y_pred (np. array): Predicted values
            dates (np.array): Date values
            figsize (tuple): Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if dates is not None:
            ax.plot(dates, y_test, label='Actual Price', linewidth=2, alpha=0.8)
            ax.plot(dates, y_pred, label='Predicted Price', linewidth=2, alpha=0.8)
        else:
            ax.plot(y_test, label='Actual Price', linewidth=2, alpha=0.8)
            ax.plot(y_pred, label='Predicted Price', linewidth=2, alpha=0.8)
        
        ax.set_title('Actual vs Predicted Stock Prices', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time' if dates is None else 'Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt. tight_layout()
        return fig
    
    @staticmethod
    def plot_prediction_error(y_test, y_pred, figsize=(14, 5)):
        """
        Plot prediction errors
        
        Args:
            y_test (np.array): Actual values
            y_pred (np.array): Predicted values
            figsize (tuple): Figure size
        """
        errors = y_test. flatten() - y_pred.flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Error over time
        ax1.plot(errors, linewidth=1)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample', fontsize=12)
        ax1.set_ylabel('Error ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Error ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_technical_indicators(df, figsize=(14, 12)):
        """
        Plot technical indicators
        
        Args: 
            df (pd.DataFrame): Stock data with indicators
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Price with Moving Averages
        axes[0]. plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
        if 'MA_50' in df.columns:
            axes[0].plot(df['Date'], df['MA_50'], label='MA 50', linewidth=1.5, alpha=0.7)
        if 'MA_200' in df.columns:
            axes[0].plot(df['Date'], df['MA_200'], label='MA 200', linewidth=1.5, alpha=0.7)
        axes[0].set_title('Price with Moving Averages', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in df. columns:
            axes[1]. plot(df['Date'], df['RSI'], label='RSI', linewidth=2, color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', label='Oversold')
            axes[1].set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('RSI', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in df.columns:
            axes[2].plot(df['Date'], df['MACD'], label='MACD', linewidth=2)
            axes[2].plot(df['Date'], df['MACD_Signal'], label='Signal', linewidth=2)
            axes[2].bar(df['Date'], df['MACD_Hist'], label='Histogram', alpha=0.3)
            axes[2].set_title('MACD', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Date', fontsize=12)
            axes[2].set_ylabel('MACD', fontsize=12)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_interactive_chart(df, ticker):
        """
        Create interactive plotly chart
        
        Args: 
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker
            
        Returns:
            plotly.graph_objects.Figure: Interactive figure
        """
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price & Moving Averages', 'RSI', 'Volume'),
                           row_heights=[0.5, 0.25, 0.25])
        
        # Price and MAs
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], 
                                name='Close', line=dict(width=2)),
                     row=1, col=1)
        
        if 'MA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'], 
                                    name='MA 50', line=dict(width=1)),
                         row=1, col=1)
        
        if 'MA_200' in df.columns:
            fig.add_trace(go. Scatter(x=df['Date'], y=df['MA_200'], 
                                    name='MA 200', line=dict(width=1)),
                         row=1, col=1)
        
        # RSI
        if 'RSI' in df. columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], 
                                    name='RSI', line=dict(color='purple', width=2)),
                         row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'),
                     row=3, col=1)
        
        fig.update_layout(
            title=f'{ticker} Stock Analysis',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        return fig
    
    @staticmethod
    def plot_future_predictions(historical_prices, future_predictions, 
                               historical_dates, future_dates, ticker, figsize=(14, 7)):
        """
        Plot historical prices and future predictions
        
        Args:
            historical_prices (np.array): Historical price data
            future_predictions (np.array): Future predictions
            historical_dates:  Historical dates
            future_dates: Future prediction dates
            ticker (str): Stock ticker
            figsize (tuple): Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical prices
        ax.plot(historical_dates, historical_prices, 
               label='Historical Prices', linewidth=2, color='blue')
        
        # Plot future predictions
        ax.plot(future_dates, future_predictions, 
               label='Future Predictions', linewidth=2, 
               color='red', linestyle='--', marker='o')
        
        # Add vertical line at prediction start
        ax.axvline(x=historical_dates. iloc[-1], color='gray', 
                  linestyle='--', alpha=0.7, label='Prediction Start')
        
        ax.set_title(f'{ticker} - Price Predictions', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_candlestick_chart(df, ticker, days=90):
        """
        Create interactive candlestick chart with volume
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker
            days (int): Number of days to show
        """
        df_recent = df.tail(days)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, 
                           subplot_titles=(f'{ticker} Candlestick Chart', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df_recent['Date'],
                                     open=df_recent['Open'],
                                     high=df_recent['High'],
                                     low=df_recent['Low'],
                                     close=df_recent['Close'],
                                     name='OHLC'),
                     row=1, col=1)
        
        # Add Bollinger Bands if available
        if 'BB_Upper' in df_recent.columns:
            fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['BB_Upper'],
                                    name='BB Upper', line=dict(color='gray', dash='dash', width=1)),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=df_recent['Date'], y=df_recent['BB_Lower'],
                                    name='BB Lower', line=dict(color='gray', dash='dash', width=1),
                                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
                         row=1, col=1)
        
        # Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                 for _, row in df_recent.iterrows()]
        fig.add_trace(go.Bar(x=df_recent['Date'], y=df_recent['Volume'],
                            name='Volume', marker_color=colors),
                     row=2, col=1)
        
        fig.update_layout(height=700, showlegend=True, xaxis_rangeslider_visible=False)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df, figsize=(12, 10)):
        """
        Plot correlation heatmap of features
        
        Args:
            df (pd.DataFrame): Stock data with features
            figsize (tuple): Figure size
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_prediction_intervals(y_test, y_pred, confidence=0.95, figsize=(14, 7)):
        """
        Plot predictions with confidence intervals
        
        Args:
            y_test (np.array): Actual values
            y_pred (np.array): Predicted values
            confidence (float): Confidence level
            figsize (tuple): Figure size
        """
        errors = y_test.flatten() - y_pred.flatten()
        std_error = np.std(errors)
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        upper_bound = y_pred.flatten() + z_score * std_error
        lower_bound = y_pred.flatten() - z_score * std_error
        
        fig, ax = plt.subplots(figsize=figsize)
        x = range(len(y_test))
        
        ax.plot(x, y_test.flatten(), label='Actual', linewidth=2, color='blue', alpha=0.7)
        ax.plot(x, y_pred.flatten(), label='Predicted', linewidth=2, color='red', alpha=0.7)
        ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, color='red',
                        label=f'{int(confidence*100)}% Confidence Interval')
        
        ax.set_title(f'Predictions with {int(confidence*100)}% Confidence Interval', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_residual_analysis(y_test, y_pred, figsize=(14, 10)):
        """
        Comprehensive residual analysis plots
        
        Args:
            y_test (np.array): Actual values
            y_pred (np.array): Predicted values
            figsize (tuple): Figure size
        """
        from scipy import stats
        residuals = y_test.flatten() - y_pred.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred.flatten(), residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
        axes[0, 0].set_ylabel('Residuals', fontsize=12)
        axes[0, 0].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals over time
        axes[1, 1].plot(residuals, linewidth=1)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Sample', fontsize=12)
        axes[1, 1].set_ylabel('Residuals', fontsize=12)
        axes[1, 1].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance_proxy(X_train, feature_names, figsize=(10, 8)):
        """
        Plot feature variance as a proxy for importance
        
        Args:
            X_train (np.array): Training data
            feature_names (list): Names of features
            figsize (tuple): Figure size
        """
        # Calculate variance across time steps for each feature
        variances = np.var(X_train, axis=(0, 1))
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Variance': variances
        }).sort_values('Variance', ascending=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(importance_df['Feature'], importance_df['Variance'], color='steelblue')
        ax.set_xlabel('Variance (Proxy for Importance)', fontsize=12)
        ax.set_title('Feature Importance (Variance)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_multi_model_errors(y_test, model_predictions, figsize=(16, 10)):
        """
        Plot prediction errors for multiple models in combined view
        
        Args:
            y_test (np.array): Actual values
            model_predictions (dict): Dictionary of {model_name: predictions}
            figsize (tuple): Figure size
        """
        n_models = len(model_predictions)
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Errors over time for all models
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            errors = y_test.flatten() - y_pred.flatten()
            ax1.plot(errors, label=model_name, linewidth=1.5, alpha=0.7, color=colors[idx])
        
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax1.set_title('Prediction Errors Over Time - All Models', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample', fontsize=12)
        ax1.set_ylabel('Error ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Error distributions for all models
        all_errors = []
        labels = []
        for model_name, y_pred in model_predictions.items():
            errors = y_test.flatten() - y_pred.flatten()
            all_errors.append(errors)
            labels.append(model_name)
        
        bp = ax2.boxplot(all_errors, labels=labels, patch_artist=True, showmeans=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax2.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Error ($)', fontsize=12)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_multi_model_confidence_intervals(y_test, model_predictions, confidence=0.95, figsize=(16, 10)):
        """
        Plot predictions with confidence intervals for multiple models
        
        Args:
            y_test (np.array): Actual values
            model_predictions (dict): Dictionary of {model_name: predictions}
            confidence (float): Confidence level
            figsize (tuple): Figure size
        """
        n_models = len(model_predictions)
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        fig, ax = plt.subplots(figsize=figsize)
        x = range(len(y_test))
        
        # Plot actual values
        ax.plot(x, y_test.flatten(), label='Actual', linewidth=2.5, color='black', alpha=0.8, zorder=10)
        
        # Plot predictions and confidence intervals for each model
        z_score = 1.96 if confidence == 0.95 else 2.576
        
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            errors = y_test.flatten() - y_pred.flatten()
            std_error = np.std(errors)
            
            upper_bound = y_pred.flatten() + z_score * std_error
            lower_bound = y_pred.flatten() - z_score * std_error
            
            # Plot prediction line
            ax.plot(x, y_pred.flatten(), label=model_name, linewidth=1.5, 
                   color=colors[idx], alpha=0.8, linestyle='--')
            
            # Plot confidence interval
            ax.fill_between(x, lower_bound, upper_bound, alpha=0.15, color=colors[idx])
        
        ax.set_title(f'Multi-Model Predictions with {int(confidence*100)}% Confidence Intervals', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_multi_model_residuals(y_test, model_predictions, figsize=(18, 12)):
        """
        Comprehensive residual analysis for multiple models side-by-side
        
        Args:
            y_test (np.array): Actual values
            model_predictions (dict): Dictionary of {model_name: predictions}
            figsize (tuple): Figure size
        """
        from scipy import stats
        
        n_models = len(model_predictions)
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Residuals vs Predicted for all models
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            residuals = y_test.flatten() - y_pred.flatten()
            axes[0, 0].scatter(y_pred.flatten(), residuals, alpha=0.5, 
                             label=model_name, color=colors[idx], s=20)
        
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
        axes[0, 0].set_ylabel('Residuals', fontsize=12)
        axes[0, 0].set_title('Residuals vs Predicted - All Models', fontsize=14, fontweight='bold')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Histogram of residuals overlaid
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            residuals = y_test.flatten() - y_pred.flatten()
            axes[0, 1].hist(residuals, bins=30, alpha=0.5, label=model_name, 
                          color=colors[idx], edgecolor='black', linewidth=0.5)
        
        axes[0, 1].set_xlabel('Residuals', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution of Residuals - All Models', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc='best')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Q-Q plots overlaid
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            residuals = y_test.flatten() - y_pred.flatten()
            (theoretical_quantiles, ordered_values), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
            axes[1, 0].scatter(theoretical_quantiles, ordered_values, alpha=0.5, 
                             label=model_name, color=colors[idx], s=20)
        
        # Add reference line
        xlim = axes[1, 0].get_xlim()
        axes[1, 0].plot(xlim, xlim, 'k--', linewidth=2, label='Normal')
        axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=12)
        axes[1, 0].set_ylabel('Sample Quantiles', fontsize=12)
        axes[1, 0].set_title('Q-Q Plot - All Models', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='best')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals over time
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            residuals = y_test.flatten() - y_pred.flatten()
            axes[1, 1].plot(residuals, linewidth=1.5, alpha=0.7, 
                          label=model_name, color=colors[idx])
        
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Sample', fontsize=12)
        axes[1, 1].set_ylabel('Residuals', fontsize=12)
        axes[1, 1].set_title('Residuals Over Time - All Models', fontsize=14, fontweight='bold')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_candlestick_chart(df, ticker, patterns=None, support_resistance=None, show_volume=True, figsize=(14, 10)):
        """
        Plot interactive candlestick chart with OHLC data, volume, support/resistance levels, and pattern annotations
        
        Args:
            df (pd.DataFrame): Stock data with OHLC columns
            ticker (str): Stock ticker
            patterns (dict): Dictionary of detected patterns with dates and descriptions
            support_resistance (dict): Dictionary with 'support' and 'resistance' levels
            show_volume (bool): Whether to show volume subplot
            figsize (tuple): Figure size
        """
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{ticker} Price Action', 'Volume')
            )
        else:
            fig = go.Figure()
        
        # Candlestick chart
        candlestick = go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add moving averages
        if 'MA_50' in df.columns:
            ma50 = go.Scatter(
                x=df['Date'],
                y=df['MA_50'],
                mode='lines',
                name='MA 50',
                line=dict(color='#ffa726', width=1.5)
            )
            if show_volume:
                fig.add_trace(ma50, row=1, col=1)
            else:
                fig.add_trace(ma50)
        
        if 'MA_200' in df.columns:
            ma200 = go.Scatter(
                x=df['Date'],
                y=df['MA_200'],
                mode='lines',
                name='MA 200',
                line=dict(color='#42a5f5', width=1.5)
            )
            if show_volume:
                fig.add_trace(ma200, row=1, col=1)
            else:
                fig.add_trace(ma200)
        
        # Add support/resistance levels
        if support_resistance:
            if 'support' in support_resistance:
                for level in support_resistance['support']:
                    fig.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Support: ${level:.2f}",
                        annotation_position="right",
                        row=1 if show_volume else None,
                        col=1 if show_volume else None
                    )
            
            if 'resistance' in support_resistance:
                for level in support_resistance['resistance']:
                    fig.add_hline(
                        y=level,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Resistance: ${level:.2f}",
                        annotation_position="right",
                        row=1 if show_volume else None,
                        col=1 if show_volume else None
                    )
        
        # Add pattern annotations
        if patterns:
            for date, pattern_info in patterns.items():
                # Find the price at this date
                date_data = df[df['Date'] == date]
                if not date_data.empty:
                    y_position = date_data['High'].values[0] * 1.02  # Slightly above the high
                    fig.add_annotation(
                        x=date,
                        y=y_position,
                        text=pattern_info.get('name', 'Pattern'),
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=pattern_info.get('color', '#ff6b6b'),
                        font=dict(size=10, color=pattern_info.get('color', '#ff6b6b')),
                        bgcolor='rgba(255,255,255,0.8)',
                        row=1 if show_volume else None,
                        col=1 if show_volume else None
                    )
        
        # Add volume bars
        if show_volume:
            colors = ['#26a69a' if close >= open else '#ef5350' 
                     for close, open in zip(df['Close'], df['Open'])]
            
            volume_bars = go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            )
            fig.add_trace(volume_bars, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Interactive Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=figsize[1] * 50,
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        if show_volume:
            fig.update_yaxes(title_text='Price ($)', row=1, col=1)
            fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        return fig
    
    @staticmethod
    def detect_support_resistance(df, num_levels=3, window=20):
        """
        Detect support and resistance levels using local minima and maxima
        
        Args:
            df (pd.DataFrame): Stock data
            num_levels (int): Number of support/resistance levels to detect
            window (int): Window size for local extrema detection
        
        Returns:
            dict: Dictionary with 'support' and 'resistance' lists
        """
        from scipy.signal import argrelextrema
        
        # Find local minima (support levels)
        local_min_indices = argrelextrema(df['Low'].values, np.less, order=window)[0]
        support_levels = df['Low'].iloc[local_min_indices].values
        
        # Find local maxima (resistance levels)
        local_max_indices = argrelextrema(df['High'].values, np.greater, order=window)[0]
        resistance_levels = df['High'].iloc[local_max_indices].values
        
        # Get the most significant levels (cluster similar levels)
        def cluster_levels(levels, num_clusters):
            if len(levels) == 0:
                return []
            if len(levels) <= num_clusters:
                return sorted(levels)
            
            # Simple clustering by grouping nearby levels
            sorted_levels = sorted(levels)
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < 0.02:  # 2% threshold
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            
            # Return top num_clusters by frequency/strength
            if len(clusters) > num_clusters:
                return sorted(clusters)[-num_clusters:]
            return sorted(clusters)
        
        support = cluster_levels(support_levels, num_levels)
        resistance = cluster_levels(resistance_levels, num_levels)
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    @staticmethod
    def plot_model_comparison_dashboard(comparison_results, model_predictions, y_test_actual, test_dates):
        """
        Create comprehensive model comparison dashboard with side-by-side metrics,
        error distributions, and performance over different market conditions
        
        Args:
            comparison_results (pd.DataFrame): DataFrame with model performance metrics
            model_predictions (dict): Dictionary of model predictions {model_name: predictions}
            y_test_actual (np.array): Actual test values
            test_dates (pd.Series): Dates for test period
        
        Returns:
            tuple: (metrics_fig, distribution_fig, conditions_fig)
        """
        # Figure 1: Side-by-side performance metrics
        metrics_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE Comparison', 
                          'MAE Comparison', 'MAPE Comparison'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        models = comparison_results['Model'].tolist()
        colors = ['#818cf8', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#ec4899']
        
        # R² Score
        metrics_fig.add_trace(
            go.Bar(x=models, y=comparison_results['R² Score'], 
                  marker_color=colors[:len(models)], name='R² Score'),
            row=1, col=1
        )
        
        # RMSE
        metrics_fig.add_trace(
            go.Bar(x=models, y=comparison_results['RMSE'], 
                  marker_color=colors[:len(models)], name='RMSE'),
            row=1, col=2
        )
        
        # MAE
        metrics_fig.add_trace(
            go.Bar(x=models, y=comparison_results['MAE'], 
                  marker_color=colors[:len(models)], name='MAE'),
            row=2, col=1
        )
        
        # MAPE
        metrics_fig.add_trace(
            go.Bar(x=models, y=comparison_results['MAPE (%)'], 
                  marker_color=colors[:len(models)], name='MAPE (%)'),
            row=2, col=2
        )
        
        metrics_fig.update_layout(
            height=600,
            showlegend=False,
            title_text='Model Performance Metrics Comparison',
            template='plotly_dark'
        )
        
        # Figure 2: Error distribution comparisons
        distribution_fig = go.Figure()
        
        for i, (model_name, predictions) in enumerate(model_predictions.items()):
            errors = y_test_actual.flatten() - predictions.flatten()
            
            distribution_fig.add_trace(go.Box(
                y=errors,
                name=model_name,
                marker_color=colors[i % len(colors)],
                boxmean='sd'  # Show mean and standard deviation
            ))
        
        distribution_fig.update_layout(
            title='Error Distribution Comparison Across Models',
            yaxis_title='Prediction Error ($)',
            height=500,
            template='plotly_dark',
            showlegend=True
        )
        
        # Figure 3: Performance over different market conditions
        # Calculate market volatility and categorize periods
        returns = pd.Series(y_test_actual.flatten()).pct_change().fillna(0)
        volatility = returns.rolling(window=10).std().fillna(0)
        
        # Categorize into low, medium, high volatility
        low_vol_mask = volatility < volatility.quantile(0.33)
        med_vol_mask = (volatility >= volatility.quantile(0.33)) & (volatility < volatility.quantile(0.67))
        high_vol_mask = volatility >= volatility.quantile(0.67)
        
        conditions_data = []
        for model_name, predictions in model_predictions.items():
            errors = np.abs(y_test_actual.flatten() - predictions.flatten())
            
            conditions_data.append({
                'Model': model_name,
                'Low Volatility MAE': np.mean(errors[low_vol_mask]),
                'Medium Volatility MAE': np.mean(errors[med_vol_mask]),
                'High Volatility MAE': np.mean(errors[high_vol_mask])
            })
        
        conditions_df = pd.DataFrame(conditions_data)
        
        conditions_fig = go.Figure()
        
        conditions_fig.add_trace(go.Bar(
            name='Low Volatility',
            x=conditions_df['Model'],
            y=conditions_df['Low Volatility MAE'],
            marker_color='#10b981'
        ))
        
        conditions_fig.add_trace(go.Bar(
            name='Medium Volatility',
            x=conditions_df['Model'],
            y=conditions_df['Medium Volatility MAE'],
            marker_color='#f59e0b'
        ))
        
        conditions_fig.add_trace(go.Bar(
            name='High Volatility',
            x=conditions_df['Model'],
            y=conditions_df['High Volatility MAE'],
            marker_color='#ef4444'
        ))
        
        conditions_fig.update_layout(
            title='Model Performance Across Market Conditions',
            yaxis_title='Mean Absolute Error ($)',
            barmode='group',
            height=500,
            template='plotly_dark',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return metrics_fig, distribution_fig, conditions_fig