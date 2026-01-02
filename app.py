"""
Streamlit Dashboard for Stock Price Prediction
Interactive web application for LSTM-based stock price forecasting
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import logging
import time
import io
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import StockDataFetcher
from src.feature_engineering import FeatureEngineer
from src.preprocessor import DataPreprocessor
from src.model import LSTMModel
from src.trainer import ModelTrainer
from src.nbeats_model import NBeatsModel
from src.nbeats_trainer import NBeatsTrainer
from src.attention_lstm_model import AttentionLSTMModel
from src.tcn_model import TCNModel
from src.transformer_model import TransformerModel
from src.ensemble_model import EnsembleModel
from src.visualizer import Visualizer
from src.trading_signals import TradingSignals
from src.portfolio_manager import PortfolioManager
from src.news_sentiment import NewsSentimentAnalyzer
from src.risk_metrics import RiskAnalyzer
from src.pattern_recognition import PatternRecognizer
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Initialize session state for features
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = PortfolioManager(initial_cash=100000)
if 'comparison_tickers' not in st.session_state:
    st.session_state.comparison_tickers = []
if 'comparison_done' not in st.session_state:
    st.session_state.comparison_done = False
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'comparison_histories' not in st.session_state:
    st.session_state.comparison_histories = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Custom CSS for enhanced modern styling
bg_gradient = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
text_color = "#e2e8f0"
card_bg = "linear-gradient(135deg, #1e293b 0%, #334155 100%)"

st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {{
        background: {bg_gradient};
    }}
    
    /* Ensure all text is visible with good contrast */
    .main .block-container {{
        color: {text_color};
    }}
    
    /* Fix for metric containers */
    div[data-testid="stMetric"] {{
        background: {card_bg};
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 2px solid rgba(129, 140, 248, 0.4);
        transition: all 0.3s ease;
    }}
    
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(129, 140, 248, 0.4);
    }}
    
    /* Main Header */
    .main-header {{
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
        animation: fadeInDown 1s ease-in-out;
    }}
    
    .sub-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
        border-bottom: 3px solid #818cf8;
        padding-bottom: 0.5rem;
    }}
    
    /* Info Boxes */
    .info-box {{
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-left: 6px solid #60a5fa;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        color: #e0f2fe !important;
    }}
    
    .info-box h4 {{
        color: #bfdbfe !important;
        margin: 0 !important;
        font-weight: 600 !important;
    }}
    
    .info-box p {{
        color: #dbeafe !important;
        margin: 0.5rem 0 0 0 !important;
    }}
    
    .info-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }}
    
    .warning-box {{
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        border-left: 6px solid #fbbf24;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        color: #fef3c7 !important;
    }}
    
    .warning-box h4 {{
        color: #fde68a !important;
        margin: 0 !important;
        font-weight: 600 !important;
    }}
    
    .warning-box p {{
        color: #fef3c7 !important;
        margin: 0.5rem 0 0 0 !important;
    }}
    
    .warning-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }}
    
    .success-box {{
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-left: 6px solid #34d399;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        color: #d1fae5 !important;
    }}
    
    .success-box h4 {{
        color: #a7f3d0 !important;
        margin: 0 !important;
        font-weight: 600 !important;
    }}
    
    .success-box p {{
        color: #d1fae5 !important;
        margin: 0.5rem 0 0 0 !important;
    }}
    
    .success-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid rgba(129, 140, 248, 0.4);
    }}
    
    .metric-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(129, 140, 248, 0.4);
    }}
    
    .metric-card [data-testid="stMetricLabel"] {{
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        width: 100%;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(129, 140, 248, 0.4);
        font-family: 'Poppins', sans-serif;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(129, 140, 248, 0.6);
        background: linear-gradient(135deg, #a78bfa 0%, #818cf8 100%);
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: #e2e8f0;
    }}
    
    /* Dataframe Styling */
    .dataframe {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        border: 2px solid #64748b;
        transition: all 0.3s ease;
        color: #e2e8f0 !important;
        font-size: 1rem;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%) !important;
        color: white !important;
        border: 2px solid #818cf8;
    }}
    
    /* Animations */
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Metric Value Styling */
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #818cf8 !important;
        font-family: 'Poppins', sans-serif !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    [data-testid="stMetricDelta"] {{
        font-size: 1rem !important;
        font-weight: 600 !important;
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        border-radius: 12px;
        font-weight: 600;
        color: #e2e8f0;
    }}
    
    /* Download Button */
    .stDownloadButton>button {{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
    }}
    
    .stDownloadButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }}
    
    /* Code Block Styling */
    .stCodeBlock {{
        background-color: #1e293b !important;
        border-radius: 12px;
        border: 2px solid #334155;
    }}
    
    /* Info/Warning Messages */
    .stAlert {{
        border-radius: 12px;
        border-left-width: 5px;
    }}
    
    /* Ensure proper text contrast */
    .element-container, .stMarkdown, p, span, div {{
        color: inherit;
    }}
    
    /* Default heading colors - will be overridden by specific contexts */
    h1, h2, h3, h4, h5, h6 {{
        color: #e2e8f0;
    }}
    
    /* Light headings on main content area */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
        color: #e2e8f0 !important;
    }}
    
    /* Section titles styling */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {{
        color: #e2e8f0 !important;
    }}
</style>
""", unsafe_allow_html=True)

# Title with enhanced subtitle
st.markdown('<h1 class="main-header">📈 Stock Price Predictor with LSTM</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #94a3b8; font-family: Poppins, sans-serif;'>
        Harness the power of Deep Learning to predict stock market trends with precision
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Configuration
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; margin-bottom: 1rem;'>
    <h1 style='font-size: 3rem; margin: 0;'>📈</h1>
</div>
""", unsafe_allow_html=True)
st.sidebar.title("⚙️ Configuration")

# Auto-refresh
st.sidebar.checkbox(
    "🔄 Auto-Refresh Data",
    value=st.session_state.auto_refresh,
    key='auto_refresh',
    help="Automatically refresh data every 5 minutes"
)

st.sidebar.markdown("---")

# Stock Selection
st.sidebar.subheader("📊 Stock Selection")
ticker = st.sidebar.text_input(
    "Stock Ticker Symbol", 
    value=config.DEFAULT_TICKER,
    help="Enter a valid stock ticker (e.g., AAPL, GOOGL, MSFT, TSLA)"
).upper()

# Popular stocks quick select - diversified across sectors
popular_stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Meta": "META",
    "Netflix": "NFLX",
    "NVIDIA": "NVDA",
    "Coca-Cola": "KO",
    "Johnson & Johnson": "JNJ",
    "Walmart": "WMT",
    "Procter & Gamble": "PG",
    "JPMorgan Chase": "JPM",
    "Bank of America": "BAC",
    "Visa": "V",
    "Mastercard": "MA",
    "Disney": "DIS",
    "McDonald's": "MCD",
    "Nike": "NKE",
    "Exxon Mobil": "XOM",
    "Chevron": "CVX",
    "Pfizer": "PFE",
    "UnitedHealth": "UNH",
    "Home Depot": "HD"
}

selected_stock = st.sidebar.selectbox(
    "Or select from popular stocks:",
    options=["Custom"] + list(popular_stocks. keys())
)

if selected_stock != "Custom":
    ticker = popular_stocks[selected_stock]

st.sidebar.markdown("---")

# Date Range Selection
st.sidebar.subheader("📅 Data Period")
period_options = {
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Maximum": "max"
}
period_label = st.sidebar.selectbox(
    "Select time period",
    options=list(period_options.keys()),
    index=2
)
period = period_options[period_label]

st.sidebar.markdown("---")

# Model Parameters
st.sidebar.subheader("🤖 Model Parameters")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    options=["LSTM", "Attention-LSTM", "N-BEATS", "TCN", "Transformer", "Ensemble"],
    index=0,
    help="""Choose a model architecture:
    - LSTM: Recurrent neural network with memory cells
    - Attention-LSTM: LSTM with attention mechanism for feature importance
    - N-BEATS: Feed-forward with interpretable trend & seasonality
    - TCN: Temporal Convolutional Network with dilated convolutions
    - Transformer: Multi-head self-attention architecture
    - Ensemble: Combines predictions from multiple models
    """
)

seq_length = st.sidebar.slider(
    "Sequence Length (days)",
    min_value=30,
    max_value=90,
    value=config.SEQ_LENGTH,
    step=10,
    help="Number of past days to use for prediction"
)

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=10,
    max_value=100,
    value=config.EPOCHS,
    step=10,
    help="Number of training iterations"
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    options=[16, 32, 64, 128],
    index=1,
    help="Number of samples per training batch"
)

# Train all models option
train_all_models = st.sidebar.checkbox(
    "🔥 Train All Models & Compare",
    value=False,
    help="Train all 5 individual models and compare their performance"
)

st.sidebar.markdown("---")

# Future Predictions
st.sidebar.subheader("🔮 Future Predictions")
future_days = st.sidebar.slider(
    "Days to Predict",
    min_value=7,
    max_value=90,
    value=30,
    step=7,
    help="Number of days to forecast into the future"
)

st.sidebar.markdown("---")

# Action Buttons
train_model = st.sidebar.button("🚀 Train Model", type="primary", help="Start model training")
reset_app = st.sidebar.button("🔄 Reset Application", help="Clear all data and start over")

# Reset functionality
if reset_app:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize session state
if 'model_trained' not in st.session_state:
    # Check if saved models exist
    lstm_model_exists = os.path.exists(config.MODEL_PATH)
    attention_lstm_exists = os.path.exists('models/attention_lstm_model.pth')
    nbeats_model_exists = os.path.exists('models/nbeats_model.pth')
    tcn_model_exists = os.path.exists('models/tcn_model.pth')
    transformer_model_exists = os.path.exists('models/transformer_model.pth')
    scalers_exist = (os.path.exists('models/feature_scaler.pkl') and 
                     os.path.exists('models/target_scaler.pkl'))
    
    # Delete old Keras .h5 models if they exist (migration to PyTorch)
    old_keras_models = ['models/lstm_stock_model.h5', 'models/best_model.h5']
    for old_model in old_keras_models:
        if os.path.exists(old_model):
            try:
                os.remove(old_model)
                logger.info(f"Removed old Keras model: {old_model}")
            except Exception as e:
                logger.warning(f"Could not remove old model {old_model}: {e}")
    
    # Try to load LSTM model if it exists
    if lstm_model_exists and scalers_exist:
        try:
            # Load the saved model and scalers
            from src.model import LSTMModel
            from src.preprocessor import DataPreprocessor
            from src.trainer import ModelTrainer
            
            # Create preprocessor and load scalers
            preprocessor = DataPreprocessor(
                feature_columns=config.FEATURES,
                target_column=config.TARGET
            )
            preprocessor.load_scalers()
            
            # Create LSTM model instance and load the trained model
            lstm_model = LSTMModel(
                seq_length=seq_length,
                n_features=len(config.FEATURES),
                lstm_units=config.LSTM_UNITS,
                dropout_rate=config.DROPOUT_RATE,
                learning_rate=config.LEARNING_RATE
            )
            lstm_model.load_model_from_file(config.MODEL_PATH)
            
            # Create trainer with loaded model
            trainer = ModelTrainer(lstm_model)
            
            # Store in session state
            st.session_state.model_trained = True
            st.session_state.lstm_model = lstm_model
            st.session_state.trainer = trainer
            st.session_state.preprocessor = preprocessor
            st.session_state.model_type = 'LSTM'
            
            logger.info("Successfully loaded saved LSTM model and scalers from previous session")
        except Exception as e:
            logger.warning(f"Failed to load saved LSTM model: {e}")
            st.session_state.model_trained = False
    
    # Try to load other models if they exist
    if scalers_exist:
        # Load preprocessor once for all models
        try:
            preprocessor = DataPreprocessor(
                feature_columns=config.FEATURES,
                target_column=config.TARGET
            )
            preprocessor.load_scalers()
            
            # Load Attention-LSTM if exists
            if attention_lstm_exists:
                try:
                    attention_lstm_model = AttentionLSTMModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        lstm_units=[64, 64],
                        dropout_rate=config.DROPOUT_RATE,
                        learning_rate=config.LEARNING_RATE
                    )
                    attention_lstm_model.load_model_from_file('models/attention_lstm_model.pth')
                    st.session_state.attention_lstm_model = attention_lstm_model
                    st.session_state.model_trained = True
                    logger.info("Loaded Attention-LSTM model")
                except Exception as e:
                    logger.warning(f"Failed to load Attention-LSTM model: {e}")
            
            # Load TCN if exists
            if tcn_model_exists:
                try:
                    tcn_model = TCNModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        num_channels=[32, 32, 64, 64],
                        kernel_size=3,
                        dropout_rate=config.DROPOUT_RATE,
                        learning_rate=config.LEARNING_RATE
                    )
                    tcn_model.load_model_from_file('models/tcn_model.pth')
                    st.session_state.tcn_model = tcn_model
                    st.session_state.model_trained = True
                    logger.info("Loaded TCN model")
                except Exception as e:
                    logger.warning(f"Failed to load TCN model: {e}")
            
            # Load Transformer if exists
            if transformer_model_exists:
                try:
                    transformer_model = TransformerModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        d_model=64,
                        nhead=4,
                        num_layers=3,
                        dropout_rate=config.DROPOUT_RATE,
                        learning_rate=config.LEARNING_RATE
                    )
                    transformer_model.load_model_from_file('models/transformer_model.pth')
                    st.session_state.transformer_model = transformer_model
                    st.session_state.model_trained = True
                    logger.info("Loaded Transformer model")
                except Exception as e:
                    logger.warning(f"Failed to load Transformer model: {e}")
            
            # Load N-BEATS if exists
            if nbeats_model_exists:
                try:
                    nbeats_model = NBeatsModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        forecast_length=1,
                        hidden_layer_units=256,
                        stack_types=('trend', 'seasonality', 'generic'),
                        nb_blocks_per_stack=4,
                        learning_rate=0.0005
                    )
                    nbeats_model.load_model_from_file('models/nbeats_model.pth')
                    st.session_state.nbeats_model = nbeats_model
                    st.session_state.model_trained = True
                    logger.info("Loaded N-BEATS model")
                except Exception as e:
                    logger.warning(f"Failed to load N-BEATS model: {e}")
            
            # Store preprocessor if any model was loaded
            if st.session_state.get('model_trained', False):
                st.session_state.preprocessor = preprocessor
                
        except Exception as e:
            logger.warning(f"Failed to load preprocessor: {e}")
    
    # Set to False if no models were loaded
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'ticker' not in st.session_state:
    st.session_state.ticker = ticker
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# Check if ticker changed
if st.session_state.ticker != ticker:
    st.session_state.data_loaded = False
    st. session_state.model_trained = False
    st. session_state.ticker = ticker

# Auto-refresh functionality
if st.session_state.auto_refresh:
    # Initialize last refresh time if not exists
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    # Check if 5 minutes have passed
    current_time = time.time()
    time_elapsed = current_time - st.session_state.last_refresh_time
    time_remaining = max(0, 300 - time_elapsed)  # 300 seconds = 5 minutes
    
    if time_elapsed >= 300:
        # Reset the timer and clear cache to force data reload
        st.session_state.last_refresh_time = current_time
        st.cache_data.clear()
        st.rerun()
    else:
        # Show countdown without blocking
        minutes_remaining = int(time_remaining // 60)
        seconds_remaining = int(time_remaining % 60)
        st.info(f"🔄 Auto-refresh enabled - Refresh in Every {minutes_remaining} miniutes ")

# Create sidebar for tab navigation
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📑 Navigation")
    
    # Tab selection using radio buttons
    selected_tab = st.radio(
        "Select Section:",
        [
            "📊 Data & Analysis",
            "🤖 Model Training",
            "📈 Predictions",
            "🔮 Future Forecast",
            "💡 Trading Signals",
            "📊 Stock Comparison",
            "💼 Portfolio Tracker",
            "📡 Real-time Monitor",
            "⚠️ Risk Metrics",
            "🔍 Pattern Recognition",
            "📋 Reports & Export"
        ],
        key="tab_selector"
    )
    
    st.markdown("---")
    
    # Information with improved styling
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        <div style='color: #e2e8f0;'>
        <h4 style='color: #667eea;'>🤖 Stock Price Predictor</h4>
        
        This advanced application leverages <b>LSTM (Long Short-Term Memory)</b> neural networks 
        to forecast stock prices using historical data and technical indicators.
        
        <h5 style='color: #10b981; margin-top: 1rem;'>✨ Key Features:</h5>
        <ul style='line-height: 1.8;'>
            <li>📡 Real-time data from Yahoo Finance</li>
            <li>📊 Technical indicators (MA, RSI, MACD)</li>
            <li>🧠 Deep learning predictions</li>
            <li>📈 Interactive visualizations</li>
            <li>🔮 Future price forecasts</li>
        </ul>
        
        <p style='color: #f59e0b; margin-top: 1rem; font-weight: 600;'>
        ⚠️ Disclaimer: Educational purposes only. Not financial advice.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📚 Quick Start Guide", expanded=False):
        st.markdown("""
        <div style='color: #e2e8f0;'>
        <ol style='line-height: 2;'>
            <li><b style='color: #667eea;'>Select</b> a stock ticker symbol</li>
            <li><b style='color: #667eea;'>Choose</b> your preferred data period</li>
            <li><b style='color: #667eea;'>Configure</b> model parameters</li>
            <li><b style='color: #667eea;'>Train</b> the LSTM model</li>
            <li><b style='color: #667eea;'>Analyze</b> predictions & forecasts</li>
            <li><b style='color: #667eea;'>Download</b> comprehensive reports</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# Map selection to tab index
tab_map = {
    "📊 Data & Analysis": 1,
    "🤖 Model Training": 2,
    "📈 Predictions": 3,
    "🔮 Future Forecast": 4,
    "💡 Trading Signals": 5,
    "📊 Stock Comparison": 6,
    "💼 Portfolio Tracker": 7,
    "📡 Real-time Monitor": 8,
    "⚠️ Risk Metrics": 9,
    "🔍 Pattern Recognition": 10,
    "📋 Reports & Export": 11
}

active_tab = tab_map[selected_tab]

# Helper Functions
@st.cache_data(show_spinner=False)
def load_data(ticker, period):
    """Load stock data with caching"""
    try:
        fetcher = StockDataFetcher()
        df = fetcher.fetch_stock_data(ticker, period=period)
        stock_info = fetcher.get_stock_info(ticker)
        return df, stock_info, None
    except Exception as e: 
        return None, None, str(e)

@st.cache_data(show_spinner=False)
def add_features(df):
    """Add technical indicators with caching"""
    try: 
        engineer = FeatureEngineer()
        df = engineer.add_all_indicators(
            df,
            ma_windows=config.MA_WINDOWS,
            rsi_period=config.RSI_PERIOD
        )
        return df, None
    except Exception as e:
        return None, str(e)

def format_number(num):
    """Format large numbers"""
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    else:
        return f"${num: ,.2f}"

# ==================== TAB 1: DATA & ANALYSIS ====================
if active_tab == 1:
    st. markdown('<h2 class="sub-header">Stock Data & Technical Analysis</h2>', unsafe_allow_html=True)
    
    try:
        with st.spinner(f"🔄 Loading data for {ticker}... "):
            df, stock_info, error = load_data(ticker, period)
            
            if error:
                st.error(f"❌ Error loading data: {error}")
                st.stop()
            
            df, error = add_features(df)
            
            if error:
                st.error(f"❌ Error adding features: {error}")
                st.stop()
            
            st.session_state.data_loaded = True
            st.session_state.df = df
            st.session_state.stock_info = stock_info
        
        st.markdown(f"""
        <div class='success-box'>
            <h4 style='margin: 0; color: #047857;'>✅ Data Loaded Successfully</h4>
            <p style='margin: 0.5rem 0 0 0; color: #047857;'>Retrieved <b>{len(df)}</b> days of historical data for <b>{ticker}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stock Information Cards with enhanced styling
        st.markdown("### 📋 Stock Information")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Company", stock_info.get('name', ticker)[:20])
        
        with col2:
            current_price = stock_info.get('current_price', 'N/A')
            if current_price != 'N/A':
                st.metric("Current Price", f"${current_price:.2f}")
            else:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        
        with col3:
            st.metric("Sector", stock_info.get('sector', 'N/A'))
        
        with col4:
            market_cap = stock_info.get('market_cap', 'N/A')
            if market_cap != 'N/A': 
                st.metric("Market Cap", format_number(market_cap))
            else:
                st.metric("Market Cap", "N/A")
        
        with col5:
            pe_ratio = stock_info.get('pe_ratio', 'N/A')
            if pe_ratio != 'N/A': 
                st.metric("P/E Ratio", f"{pe_ratio:.2f}")
            else:
                st.metric("P/E Ratio", "N/A")
        
        st.markdown("---")
        
        # Quick Statistics
        st.markdown("### 📊 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = df['Close'].iloc[-1] - df['Close']. iloc[-2]
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
            st. metric(
                "Daily Change",
                f"${price_change:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        
        with col2:
            week_high = df['High'].tail(7).max()
            st.metric("7-Day High", f"${week_high:.2f}")
        
        with col3:
            week_low = df['Low']. tail(7).min()
            st.metric("7-Day Low", f"${week_low:.2f}")
        
        with col4:
            avg_volume = df['Volume'].tail(30).mean()
            st.metric("Avg Volume (30d)", f"{avg_volume/1e6:.2f}M")
        
        st.markdown("---")
        
        # Recent Data
        st.markdown("### 📅 Recent Data (Last 10 Days)")
        display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'RSI']
        recent_data = df[display_cols].tail(10).copy()
        recent_data['Date'] = pd.to_datetime(recent_data['Date']).dt.strftime('%Y-%m-%d')
        
        # Format numbers
        for col in ['Open', 'High', 'Low', 'Close', 'MA_50', 'MA_200']: 
            recent_data[col] = recent_data[col].apply(lambda x: f"${x:.2f}")
        recent_data['Volume'] = recent_data['Volume'].apply(lambda x: f"{x/1e6:.2f}M")
        recent_data['RSI'] = recent_data['RSI'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(recent_data, width="stretch", hide_index=True)
        
        st.markdown("---")
        
        # Statistical Summary
        st.markdown("### 📈 Statistical Summary")
        summary_cols = ['Close', 'Volume', 'MA_50', 'MA_200', 'RSI']
        summary_df = df[summary_cols].describe()
        st.dataframe(summary_df, width="stretch")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📉 Price History")
        with st.spinner("Generating price chart..."):
            fig = Visualizer.plot_price_history(df, ticker)
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.markdown("### 📊 Technical Indicators")
        with st.spinner("Generating indicator charts..."):
            fig = Visualizer.plot_technical_indicators(df)
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Interactive Chart")
        with st.spinner("Generating interactive chart..."):
            fig = Visualizer.plot_interactive_chart(df, ticker)
            st.plotly_chart(fig, width="stretch")
        
        st.markdown("---")
        
        # Correlation Heatmap
        st.markdown("### 🔥 Feature Correlation Analysis")
        with st.expander("View Correlation Heatmap"):
            with st.spinner("Generating correlation heatmap..."):
                fig = Visualizer.plot_correlation_heatmap(df)
                st.pyplot(fig)
        
        # Technical Analysis Summary
        st.markdown("---")
        st.markdown("### 🔍 Technical Analysis Summary")
        
        ma_50 = df['MA_50']. iloc[-1]
        ma_200 = df['MA_200'].iloc[-1]
        current_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD']. iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        
        col1, col2 = st. columns(2)
        
        with col1:
            st.markdown("#### Moving Averages")
            if ma_50 > ma_200:
                st. success("🟢 **Golden Cross** - Bullish Signal")
                st.write(f"MA(50): ${ma_50:.2f} > MA(200): ${ma_200:.2f}")
            else:
                st.error("🔴 **Death Cross** - Bearish Signal")
                st.write(f"MA(50): ${ma_50:.2f} < MA(200): ${ma_200:.2f}")
            
            if current_price > ma_50:
                st.info("Price is above MA(50) - Short-term bullish")
            else:
                st.warning("Price is below MA(50) - Short-term bearish")
        
        with col2:
            st.markdown("#### RSI & MACD")
            if rsi > 70:
                st.markdown(f"""
                <div class='warning-box'>
                    <h4 style='margin: 0; color: #92400e;'>⚠️ Overbought Territory</h4>
                    <p style='margin: 0.5rem 0 0 0;'>RSI: <b>{rsi:.2f}</b> - Potential reversal signal</p>
                </div>
                """, unsafe_allow_html=True)
            elif rsi < 30:
                st.markdown(f"""
                <div class='success-box'>
                    <h4 style='margin: 0; color: #065f46;'>✅ Oversold Territory</h4>
                    <p style='margin: 0.5rem 0 0 0;'>RSI: <b>{rsi:.2f}</b> - Potential buying opportunity</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='info-box'>
                    <h4 style='margin: 0; color: #075985;'>📊 Neutral Zone</h4>
                    <p style='margin: 0.5rem 0 0 0;'>RSI: <b>{rsi:.2f}</b> - Balanced market conditions</p>
                </div>
                """, unsafe_allow_html=True)
            
            if macd > macd_signal:
                st.markdown("""
                <div class='success-box'>
                    <h4 style='margin: 0; color: #065f46;'>🟢 Bullish MACD Signal</h4>
                    <p style='margin: 0.5rem 0 0 0;'>MACD above Signal line - Positive momentum</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='warning-box'>
                    <h4 style='margin: 0; color: #92400e;'>🔴 Bearish MACD Signal</h4>
                    <p style='margin: 0.5rem 0 0 0;'>MACD below Signal line - Negative momentum</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interactive Candlestick Chart
        st.markdown("### 📊 Interactive Candlestick Chart")
        st.info("💡 Hover over the chart for detailed OHLC data. Includes volume, moving averages, and support/resistance levels.")
        
        with st.spinner("Detecting support/resistance levels and generating candlestick chart..."):
            # Detect support and resistance levels
            support_resistance = Visualizer.detect_support_resistance(df, num_levels=3, window=20)
            
            # Get pattern recognition if available
            patterns = None
            try:
                pattern_recognizer = PatternRecognizer()
                pattern_results = pattern_recognizer.detect_all_patterns(df)
                
                # Convert pattern results to annotations format
                patterns = {}
                for pattern_name, pattern_info in pattern_results['patterns'].items():
                    if pattern_info['last_occurrence'] is not None:
                        patterns[pattern_info['last_occurrence']] = {
                            'name': pattern_name,
                            'color': '#10b981' if pattern_info.get('bullish', False) else '#ef4444'
                        }
            except Exception as e:
                logger.warning(f"Pattern recognition failed: {e}")
            
            # Plot candlestick chart
            candlestick_fig = Visualizer.plot_candlestick_chart(
                df.tail(200),  # Last 200 days for better visibility
                ticker,
                patterns=patterns,
                support_resistance=support_resistance,
                show_volume=True
            )
            st.plotly_chart(candlestick_fig, width='stretch')
        
        # Show detected levels
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 Support Levels")
            if support_resistance['support']:
                for i, level in enumerate(support_resistance['support'], 1):
                    st.metric(f"Support {i}", f"${level:.2f}")
            else:
                st.info("No support levels detected")
        
        with col2:
            st.markdown("#### 🔴 Resistance Levels")
            if support_resistance['resistance']:
                for i, level in enumerate(support_resistance['resistance'], 1):
                    st.metric(f"Resistance {i}", f"${level:.2f}")
            else:
                st.info("No resistance levels detected")
        
        # Download data
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset",
            data=csv,
            file_name=f"{ticker}_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width='stretch'
        )
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        logger.error(f"Error in Tab 1: {str(e)}", exc_info=True)
        st.stop()

# ==================== TAB 2: MODEL TRAINING ====================
if active_tab == 2:
    st.markdown('<h2 class="sub-header">Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data in the **Data & Analysis** tab first.")
        st.stop()
    
    # Display model configuration
    st.markdown("### ⚙️ Current Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Model Parameters:**
        - Model Type: {model_type}
        - Sequence Length: {seq_length}
        - Hidden Units: {128 if model_type == 'N-BEATS' else config.LSTM_UNITS}
        - Dropout Rate: {config.DROPOUT_RATE if model_type == 'LSTM' else 'N/A'}
        """)
    
    with col2:
        st.info(f"""
        **Training Parameters:**
        - Epochs: {epochs}
        - Batch Size:  {batch_size}
        - Learning Rate: {config.LEARNING_RATE}
        """)
    
    with col3:
        st.info(f"""
        **Data Split:**
        - Training: {(1-config.TEST_SIZE)*100:.0f}%
        - Testing: {config.TEST_SIZE*100:.0f}%
        - Validation: {config. VALIDATION_SPLIT*100:.0f}%
        """)
    
    st.markdown("---")
    
    # Train all models comparison
    if train_all_models:
        st.markdown("### 🔥 Train All Models - Performance Comparison")
        st.info("📊 This will train all 5 individual models (LSTM, Attention-LSTM, N-BEATS, TCN, Transformer) and compare their performance. Ensemble is excluded to save time.")
        
        if st.button("🚀 Start Training All Models", key="train_all_btn"):
            try:
                # Prepare data once
                st.markdown("### 📊 Step 1: Preparing Data")
                progress_bar = st.progress(0)
                
                preprocessor = DataPreprocessor(config.FEATURES, config.TARGET)
                features, target, _ = preprocessor.normalize_data(st.session_state.df)
                X, y = preprocessor.create_sequences(features, target, seq_length)
                X_train, X_test, y_train, y_test = preprocessor.train_test_split(
                    X, y, test_size=config.TEST_SIZE
                )
                
                st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
                progress_bar.progress(10)
                
                # Models to train
                models_to_train = [
                    ('LSTM', LSTMModel, ModelTrainer, {'lstm_units': config.LSTM_UNITS, 'dropout_rate': config.DROPOUT_RATE}),
                    ('Attention-LSTM', AttentionLSTMModel, ModelTrainer, {'lstm_units': [64, 64], 'dropout_rate': config.DROPOUT_RATE}),
                    ('TCN', TCNModel, ModelTrainer, {'num_channels': [32, 32, 64, 64], 'kernel_size': 3, 'dropout_rate': config.DROPOUT_RATE}),
                    ('Transformer', TransformerModel, ModelTrainer, {'d_model': 64, 'nhead': 4, 'num_layers': 3, 'dropout_rate': config.DROPOUT_RATE}),
                    ('N-BEATS', NBeatsModel, NBeatsTrainer, {'forecast_length': 1, 'hidden_layer_units': 256, 'stack_types': ('trend', 'seasonality', 'generic'), 'nb_blocks_per_stack': 4, 'learning_rate': 0.0005})
                ]
                
                all_results = []
                all_histories = {}
                all_trained_models = {}  # Store trained models
                
                # Train each model
                for i, (model_name, ModelClass, TrainerClass, model_params) in enumerate(models_to_train):
                    st.markdown(f"### 🏗️ Step {i+2}: Training {model_name} Model ({i+1}/5)")
                    start_time = time.time()
                    
                    # Build model
                    if model_name == 'N-BEATS':
                        model_instance = ModelClass(
                            seq_length=seq_length,
                            n_features=len(config.FEATURES),
                            **model_params
                        )
                    else:
                        model_instance = ModelClass(
                            seq_length=seq_length,
                            n_features=len(config.FEATURES),
                            learning_rate=config.LEARNING_RATE,
                            **model_params
                        )
                    
                    model_instance.build_model()
                    trainer = TrainerClass(model_instance)
                    
                    # Train model
                    with st.spinner(f"Training {model_name}... (this may take a few minutes)"):
                        if model_name == 'N-BEATS':
                            history = trainer.train(
                                X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=config.VALIDATION_SPLIT,
                                patience=10
                            )
                        else:
                            callbacks = model_instance.get_callbacks(
                                model_path=f'models/compare_{model_name.lower().replace("-", "_")}.pth',
                                patience=10
                            )
                            history = trainer.train(
                                X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=config.VALIDATION_SPLIT,
                                callbacks=callbacks
                            )
                    
                    # Evaluate model
                    metrics = trainer.evaluate(X_test, y_test)
                    training_time = time.time() - start_time
                    
                    # Store trained model instance
                    all_trained_models[model_name] = model_instance
                    
                    # Store results
                    all_results.append({
                        'Model': model_name,
                        'R² Score': metrics['r2'],
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'MAPE (%)': metrics.get('mape', 0),
                        'Training Time (min)': training_time / 60
                    })
                    all_histories[model_name] = history
                    
                    st.success(f"✅ {model_name} completed - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, Training Time: {training_time/60:.1f} min")
                    progress_bar.progress(10 + int((i + 1) * 90 / len(models_to_train)))
                
                # Save comparison results to session state
                st.session_state.comparison_results = pd.DataFrame(all_results)
                st.session_state.comparison_histories = all_histories
                st.session_state.comparison_done = True
                
                # Store each model in session state for predictions/forecast tabs
                for model_name, model_instance in all_trained_models.items():
                    if model_name == 'LSTM':
                        st.session_state.lstm_model = model_instance
                    elif model_name == 'Attention-LSTM':
                        st.session_state.attention_lstm_model = model_instance
                    elif model_name == 'N-BEATS':
                        st.session_state.nbeats_model = model_instance
                    elif model_name == 'TCN':
                        st.session_state.tcn_model = model_instance
                    elif model_name == 'Transformer':
                        st.session_state.transformer_model = model_instance
                
                # Mark all models as trained
                st.session_state.model_trained = True
                st.session_state.preprocessor = preprocessor
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                
                # Generate predictions for dashboard
                st.markdown("### 📊 Generating predictions for comparison dashboard...")
                model_predictions = {}
                test_dates = st.session_state.df['Date'].iloc[-len(y_test):].reset_index(drop=True)
                
                for model_name, model_instance in all_trained_models.items():
                    trainer = NBeatsTrainer(model_instance) if model_name == 'N-BEATS' else ModelTrainer(model_instance)
                    y_pred = trainer.predict(X_test)
                    y_pred_actual = preprocessor.inverse_transform_target(y_pred)
                    model_predictions[model_name] = y_pred_actual
                
                # Store for dashboard
                st.session_state.comparison_predictions = model_predictions
                st.session_state.comparison_y_test = preprocessor.inverse_transform_target(y_test)
                st.session_state.comparison_test_dates = test_dates
                
                st.balloons()
                st.success("🎉 All models trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                logger.error(f"Training comparison error: {str(e)}", exc_info=True)
    
    # Display comparison results
    if st.session_state.get('comparison_done', False):
        st.markdown("---")
        st.markdown("### 🏗️ Model Architectures - All Trained Models")
        
        # Display architecture for each trained model
        model_map = {
            'LSTM': 'lstm_model',
            'Attention-LSTM': 'attention_lstm_model',
            'N-BEATS': 'nbeats_model',
            'TCN': 'tcn_model',
            'Transformer': 'transformer_model'
        }
        
        for model_name, session_key in model_map.items():
            if session_key in st.session_state:
                with st.expander(f"📐 {model_name} Architecture", expanded=False):
                    model_summary = st.session_state[session_key].get_model_summary()
                    st.code(model_summary, language='text')
        
        st.markdown("---")
        st.markdown("### 📈 Training History - All Models")
        
        if 'comparison_histories' in st.session_state and st.session_state.comparison_histories:
            for model_name, history in st.session_state.comparison_histories.items():
                with st.expander(f"📊 {model_name} Training Progress", expanded=False):
                    with st.spinner(f"Generating {model_name} training plots..."):
                        if model_name == 'N-BEATS':
                            # Create a simple history object for visualization
                            class SimpleHistory:
                                def __init__(self, history_dict):
                                    self.history = history_dict
                            
                            simple_hist = SimpleHistory({
                                'loss': history['train_loss'],
                                'val_loss': history['val_loss']
                            })
                            fig = Visualizer.plot_training_history(simple_hist)
                        else:
                            fig = Visualizer.plot_training_history(history)
                        st.pyplot(fig)
                    
                    # Training metrics table
                    if model_name == 'N-BEATS':
                        history_df = pd.DataFrame({
                            'Epoch': range(1, len(history['train_loss']) + 1),
                            'Training Loss': history['train_loss'],
                            'Validation Loss': history['val_loss'],
                        })
                    else:
                        history_df = pd.DataFrame({
                            'Epoch': range(1, len(history.history['loss']) + 1),
                            'Training Loss': history.history['loss'],
                            'Validation Loss': history.history['val_loss'],
                        })
                    
                    st.markdown(f"**{model_name} Training Metrics Summary (Last 10 Epochs)**")
                    st.dataframe(
                        history_df.tail(10).style.format({
                            'Training Loss': '{:.6f}',
                            'Validation Loss': '{:.6f}'
                        }),
                        width='stretch'
                    )
        else:
            st.info("Training histories not available. They are only stored during the current session.")
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance Comparison")
        
        results_df = st.session_state.comparison_results
        
        # Highlight best models
        st.markdown("#### 🏆 Performance Metrics")
        
        # Find best models
        best_r2_model = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
        best_rmse_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
        best_mae_model = results_df.loc[results_df['MAE'].idxmin(), 'Model']
        fastest_model = results_df.loc[results_df['Training Time (min)'].idxmin(), 'Model']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🥇 Best R² Score", best_r2_model, 
                     f"{results_df.loc[results_df['Model'] == best_r2_model, 'R² Score'].values[0]:.4f}")
        with col2:
            st.metric("🎯 Lowest RMSE", best_rmse_model,
                     f"{results_df.loc[results_df['Model'] == best_rmse_model, 'RMSE'].values[0]:.4f}")
        with col3:
            st.metric("📉 Lowest MAE", best_mae_model,
                     f"{results_df.loc[results_df['Model'] == best_mae_model, 'MAE'].values[0]:.4f}")
        with col4:
            st.metric("⚡ Fastest Training", fastest_model,
                     f"{results_df.loc[results_df['Model'] == fastest_model, 'Training Time (min)'].values[0]:.1f} min")
        
        # Display table
        st.markdown("#### 📋 Detailed Comparison Table")
        
        # Format the dataframe for display
        display_df = results_df.copy()
        display_df['R² Score'] = display_df['R² Score'].apply(lambda x: f"{x:.4f}")
        display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.4f}")
        display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.4f}")
        display_df['MAPE (%)'] = display_df['MAPE (%)'].apply(lambda x: f"{x:.2f}")
        display_df['Training Time (min)'] = display_df['Training Time (min)'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_df, width='stretch')
        
        # Visualizations
        st.markdown("#### 📈 Performance Visualizations")
        
        # Create comparison charts using plotly
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['R² Score'],
                marker_color=['#10b981' if m == best_r2_model else '#6366f1' for m in results_df['Model']],
                text=results_df['R² Score'].apply(lambda x: f"{x:.4f}"),
                textposition='outside'
            ))
            fig_r2.update_layout(
                title='R² Score Comparison (Higher is Better)',
                yaxis_title='R² Score',
                xaxis_title='Model',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_r2, width='stretch')
        
        with col2:
            # RMSE comparison
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['RMSE'],
                marker_color=['#10b981' if m == best_rmse_model else '#ef4444' for m in results_df['Model']],
                text=results_df['RMSE'].apply(lambda x: f"{x:.4f}"),
                textposition='outside'
            ))
            fig_rmse.update_layout(
                title='RMSE Comparison (Lower is Better)',
                yaxis_title='RMSE',
                xaxis_title='Model',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_rmse, width='stretch')
        
        # Training time comparison
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=results_df['Model'],
            y=results_df['Training Time (min)'],
            marker_color=['#10b981' if m == fastest_model else '#f59e0b' for m in results_df['Model']],
            text=results_df['Training Time (min)'].apply(lambda x: f"{x:.1f} min"),
            textposition='outside'
        ))
        fig_time.update_layout(
            title='Training Time Comparison',
            yaxis_title='Time (minutes)',
            xaxis_title='Model',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_time, width='stretch')
        
        # Recommendation
        st.markdown("#### 💡 Recommendation")
        
        # Simple recommendation logic
        if best_r2_model == best_rmse_model == best_mae_model:
            st.success(f"🌟 **{best_r2_model}** is the clear winner with the best accuracy across all metrics!")
        else:
            st.info(f"""
📊 **Performance Summary:**
- **Best Overall Accuracy**: {best_r2_model} (R²: {results_df.loc[results_df['Model'] == best_r2_model, 'R² Score'].values[0]:.4f})
- **Most Precise**: {best_rmse_model} (RMSE: {results_df.loc[results_df['Model'] == best_rmse_model, 'RMSE'].values[0]:.4f})
- **Fastest Training**: {fastest_model} ({results_df.loc[results_df['Model'] == fastest_model, 'Training Time (min)'].values[0]:.1f} min)

💡 **Suggestion**: Use **{best_r2_model}** for best accuracy, or **{fastest_model}** if speed is priority.
""")
        
        # Download comparison results
        st.markdown("---")
        csv_comparison = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Results (CSV)",
            data=csv_comparison,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Enhanced Model Comparison Dashboard
        st.markdown("---")
        st.markdown("### 🎯 Advanced Model Comparison Dashboard")
        st.info("📊 Comprehensive analysis of model performance including error distributions and performance across different market conditions")
        
        # Check if we have stored predictions from comparison
        if 'comparison_predictions' in st.session_state:
            model_predictions = st.session_state.comparison_predictions
            y_test_actual = st.session_state.comparison_y_test
            test_dates = st.session_state.comparison_test_dates
            
            with st.spinner("Generating comprehensive comparison dashboard..."):
                # Generate dashboard figures
                metrics_fig, distribution_fig, conditions_fig = Visualizer.plot_model_comparison_dashboard(
                    results_df,
                    model_predictions,
                    y_test_actual,
                    test_dates
                )
                
                # Display the figures
                st.markdown("#### 📊 Side-by-Side Performance Metrics")
                st.plotly_chart(metrics_fig, width='stretch')
                
                st.markdown("---")
                st.markdown("#### 📦 Error Distribution Comparison")
                st.markdown("""
                **Interpretation Guide:**
                - Boxes show the distribution of prediction errors for each model
                - The line in the middle represents the median error
                - The box boundaries show the 25th and 75th percentiles
                - Smaller boxes and errors closer to zero indicate better performance
                """)
                st.plotly_chart(distribution_fig, width='stretch')
                
                st.markdown("---")
                st.markdown("#### 🌊 Performance Across Market Conditions")
                st.markdown("""
                **Market Condition Analysis:**
                - **Low Volatility**: Stable market conditions with minimal price fluctuations
                - **Medium Volatility**: Normal market conditions with moderate price movements
                - **High Volatility**: Turbulent market conditions with significant price swings
                
                Models performing well across all conditions are more robust and reliable.
                """)
                st.plotly_chart(conditions_fig, width='stretch')
                
                # Add insights
                st.markdown("---")
                st.markdown("#### 💡 Key Insights")
                
                # Find which model performs best in each condition
                conditions_data = []
                for model_name, predictions in model_predictions.items():
                    errors = np.abs(y_test_actual.flatten() - predictions.flatten())
                    returns = pd.Series(y_test_actual.flatten()).pct_change().fillna(0)
                    volatility = returns.rolling(window=10).std().fillna(0)
                    
                    low_vol_mask = volatility < volatility.quantile(0.33)
                    med_vol_mask = (volatility >= volatility.quantile(0.33)) & (volatility < volatility.quantile(0.67))
                    high_vol_mask = volatility >= volatility.quantile(0.67)
                    
                    conditions_data.append({
                        'Model': model_name,
                        'Low Vol': np.mean(errors[low_vol_mask]),
                        'Med Vol': np.mean(errors[med_vol_mask]),
                        'High Vol': np.mean(errors[high_vol_mask])
                    })
                
                conditions_summary = pd.DataFrame(conditions_data)
                best_low_vol = conditions_summary.loc[conditions_summary['Low Vol'].idxmin(), 'Model']
                best_high_vol = conditions_summary.loc[conditions_summary['High Vol'].idxmin(), 'Model']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"🌤️ **Best in Calm Markets**\n\n{best_low_vol}")
                with col2:
                    most_consistent = conditions_summary.set_index('Model').std(axis=1).idxmin()
                    st.info(f"⚖️ **Most Consistent**\n\n{most_consistent}")
                with col3:
                    st.warning(f"🌪️ **Best in Volatility**\n\n{best_high_vol}")
        else:
            st.warning("⚠️ Advanced dashboard requires model predictions. Please run 'Train All Models & Compare' again to generate predictions.")
    
    st.markdown("---")
    
    if train_model and not train_all_models:
        try:
            # Training progress container
            progress_container = st.container()
            
            with progress_container:
                st.markdown(f"### 🚀 Training Progress - {model_type} Model")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Prepare data
                status_text.text("📊 Step 1/5: Preparing data...")
                progress_bar.progress(10)
                
                preprocessor = DataPreprocessor(config. FEATURES, config.TARGET)
                features, target, _ = preprocessor.normalize_data(st.session_state.df)
                X, y = preprocessor.create_sequences(features, target, seq_length)
                X_train, X_test, y_train, y_test = preprocessor.train_test_split(
                    X, y, test_size=config.TEST_SIZE
                )
                
                st.success(f"✅ Data prepared:  {len(X_train)} training samples, {len(X_test)} test samples")
                progress_bar.progress(20)
                
                # Step 2: Build model based on selection
                status_text.text(f"🏗️ Step 2/5: Building {model_type} model architecture...")
                progress_bar.progress(30)
                
                if model_type == "LSTM":
                    # Build LSTM model
                    model_instance = LSTMModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        lstm_units=config.LSTM_UNITS,
                        dropout_rate=config.DROPOUT_RATE,
                        learning_rate=config.LEARNING_RATE
                    )
                    model_instance.build_model()
                    trainer = ModelTrainer(model_instance)
                    st.success("✅ LSTM model architecture built")
                    
                elif model_type == "Attention-LSTM":
                    # Build Attention-LSTM model
                    model_instance = AttentionLSTMModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        lstm_units=[64, 64],  # Hidden units for each LSTM layer
                        dropout_rate=config.DROPOUT_RATE,
                        learning_rate=config.LEARNING_RATE
                    )
                    model_instance.build_model()
                    trainer = ModelTrainer(model_instance)
                    st.success("✅ Attention-LSTM model architecture built")
                    
                elif model_type == "N-BEATS":
                    # Build enhanced N-BEATS model with interpretable architecture
                    # Using trend + seasonality + generic stacks for better stock price modeling
                    model_instance = NBeatsModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        forecast_length=1,
                        hidden_layer_units=256,  # Increased from 128 for better capacity
                        stack_types=('trend', 'seasonality', 'generic'),  # Interpretable stacks
                        nb_blocks_per_stack=4,  # Increased from 3 for deeper learning
                        learning_rate=0.0005  # Lower learning rate for stability
                    )
                    model_instance.build_model()
                    trainer = NBeatsTrainer(model_instance)
                    st.success("✅ N-BEATS model architecture built")
                    
                elif model_type == "TCN":
                    # Build Temporal Convolutional Network
                    model_instance = TCNModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        num_channels=[32, 32, 64, 64],  # Channel sizes for each TCN level
                        kernel_size=3,
                        dropout_rate=config.DROPOUT_RATE,
                        learning_rate=config.LEARNING_RATE
                    )
                    model_instance.build_model()
                    trainer = ModelTrainer(model_instance)
                    st.success("✅ TCN model architecture built")
                    
                elif model_type == "Transformer":
                    # Build Transformer model
                    model_instance = TransformerModel(
                        seq_length=seq_length,
                        n_features=len(config.FEATURES),
                        d_model=64,  # Embedding dimension
                        nhead=4,  # Number of attention heads
                        num_layers=3,  # Number of transformer layers
                        dropout_rate=config.DROPOUT_RATE,
                        learning_rate=config.LEARNING_RATE
                    )
                    model_instance.build_model()
                    trainer = ModelTrainer(model_instance)
                    st.success("✅ Transformer model architecture built")
                
                elif model_type == "Ensemble":
                    # Build multiple models and ensemble them
                    models_to_ensemble = []
                    
                    # LSTM
                    lstm_model = LSTMModel(seq_length, len(config.FEATURES), config.LSTM_UNITS, 
                                         config.DROPOUT_RATE, config.LEARNING_RATE)
                    lstm_model.build_model()
                    lstm_trainer = ModelTrainer(lstm_model)
                    
                    # Attention-LSTM
                    attn_model = AttentionLSTMModel(seq_length, len(config.FEATURES), [64, 64], 
                                                    config.DROPOUT_RATE, config.LEARNING_RATE)
                    attn_model.build_model()
                    attn_trainer = ModelTrainer(attn_model)
                    
                    # TCN
                    tcn_model = TCNModel(seq_length, len(config.FEATURES), [32, 32, 64, 64], 3,
                                        config.DROPOUT_RATE, config.LEARNING_RATE)
                    tcn_model.build_model()
                    tcn_trainer = ModelTrainer(tcn_model)
                    
                    models_to_ensemble = [
                        (lstm_model, lstm_trainer),
                        (attn_model, attn_trainer),
                        (tcn_model, tcn_trainer)
                    ]
                    
                    # Create ensemble
                    model_instance = EnsembleModel(models_to_ensemble, method='weighted_average')
                    trainer = None  # Ensemble has its own prediction method
                    st.success("✅ Ensemble with 3 models (LSTM + Attention-LSTM + TCN) created")
                
                progress_bar.progress(40)
                
                # Step 3: Train model
                status_text.text(f"🎓 Step 3/5: Training {model_type} model (this may take a few minutes)...")
                
                if model_type == "Ensemble":
                    # Train each model in the ensemble
                    ensemble_history = []
                    for i, (model, trainer) in enumerate(models_to_ensemble):
                        model_name = model.__class__.__name__
                        st.info(f"Training model {i+1}/3: {model_name}")
                        
                        if isinstance(trainer, ModelTrainer):
                            callbacks = model.get_callbacks(
                                model_path=f'models/ensemble_{model_name.lower()}.pth',
                                patience=10
                            )
                            history = trainer.train(
                                X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=config.VALIDATION_SPLIT,
                                callbacks=callbacks
                            )
                        else:  # NBeatsTrainer
                            history = trainer.train(
                                X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=config.VALIDATION_SPLIT,
                                patience=10
                            )
                        ensemble_history.append(history)
                        st.success(f"✅ {model_name} trained")
                    
                    # Optimize ensemble weights
                    # Use a small validation set for weight optimization
                    val_size = int(0.1 * len(X_train))
                    X_val = X_train[-val_size:]
                    y_val = y_train[-val_size:]
                    model_instance.optimize_weights(X_val, y_val)
                    st.success("✅ Ensemble weights optimized")
                    history = ensemble_history[0]  # Use first model's history for display
                    
                elif model_type == "N-BEATS":
                    history = trainer.train(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=config.VALIDATION_SPLIT,
                        patience=10
                    )
                    st.success(f"✅ Training completed")
                    
                else:  # LSTM, Attention-LSTM, TCN, Transformer
                    callbacks = model_instance.get_callbacks(
                        model_path='models/best_model.pth',
                        patience=10
                    )
                    history = trainer.train(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=config.VALIDATION_SPLIT,
                        callbacks=callbacks
                    )
                    st.success(f"✅ Training completed after {len(history.history['loss'])} epochs")
                    
                progress_bar.progress(70)
                
                # Step 4: Evaluate model
                status_text.text("📊 Step 4/5: Evaluating model...")
                progress_bar.progress(80)
                
                if model_type == "Ensemble":
                    metrics = model_instance.evaluate(X_test, y_test)
                else:
                    metrics = trainer.evaluate(X_test, y_test)
                
                st.success("✅ Model evaluated")
                progress_bar.progress(90)
                
                # Step 5: Save model
                status_text.text("💾 Step 5/5: Saving model and scalers...")
                
                os.makedirs('models', exist_ok=True)
                if model_type == "LSTM":
                    model_instance.save_model(config.MODEL_PATH)
                elif model_type == "Attention-LSTM":
                    model_instance.save_model('models/attention_lstm_model.pth')
                elif model_type == "N-BEATS":
                    model_instance.save_model('models/nbeats_model.pth')
                elif model_type == "TCN":
                    model_instance.save_model('models/tcn_model.pth')
                elif model_type == "Transformer":
                    model_instance.save_model('models/transformer_model.pth')
                elif model_type == "Ensemble":
                    # Save each ensemble model
                    for i, (model, _) in enumerate(models_to_ensemble):
                        model_name = model.__class__.__name__
                        model.save_model(f'models/ensemble_{model_name.lower()}.pth')
                    # Save ensemble weights
                    import json
                    with open('models/ensemble_weights.json', 'w') as f:
                        json.dump({
                            'weights': model_instance.weights,
                            'method': model_instance.method,
                            'model_names': [m.__class__.__name__ for m, _ in models_to_ensemble]
                        }, f)
                    
                preprocessor.save_scalers()
                
                st.success("✅ Model and scalers saved")
                progress_bar.progress(100)
                status_text.text("🎉 Training complete!")
                
                # Store in session state
                st.session_state.model_trained = True
                st.session_state.model_type = model_type
                
                if model_type == "LSTM":
                    st.session_state.lstm_model = model_instance
                elif model_type == "Attention-LSTM":
                    st.session_state.attention_lstm_model = model_instance
                elif model_type == "N-BEATS":
                    st.session_state.nbeats_model = model_instance
                elif model_type == "TCN":
                    st.session_state.tcn_model = model_instance
                elif model_type == "Transformer":
                    st.session_state.transformer_model = model_instance
                elif model_type == "Ensemble":
                    st.session_state.ensemble_model = model_instance
                else:
                    st.session_state.nbeats_model = model_instance
                
                st.session_state.trainer = trainer
                st.session_state.preprocessor = preprocessor
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.history = history
                st.session_state.metrics = metrics
                st.session_state.training_complete = True
                
                # Debug: Confirm session state was set
                logger.info(f"Training complete. Session state updated:")
                logger.info(f"  - history type: {type(st.session_state.history)}")
                logger.info(f"  - metrics keys: {list(st.session_state.metrics.keys()) if st.session_state.metrics else 'None'}")
                
            st.balloons()
            st.markdown(f"""
            <div class='success-box' style='text-align: center;'>
                <h2 style='margin: 0; color: #065f46;'>🎊 Training Complete!</h2>
                <p style='margin: 1rem 0 0 0; font-size: 1.1rem; color: #047857;'>
                    Your {model_type} model has been successfully trained and is ready for predictions
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as e: 
            st.error(f"❌ Error during training: {str(e)}")
            logger.error(f"Training error: {str(e)}", exc_info=True)
            st.stop()
    
    # Display results if model is trained (only for single model training, not comparison mode)
    if st.session_state.model_trained and not st.session_state.get('comparison_done', False):
        st.markdown("---")
        st.markdown(f"### 🏗️ Model Architecture - {st.session_state.get('model_type', 'LSTM')}")
        
        # Display model summary based on model type
        model_type_display = st.session_state.get('model_type', 'LSTM')
        
        if model_type_display == 'LSTM':
            model_summary = st.session_state.lstm_model.get_model_summary()
            st.code(model_summary, language='text')
        elif model_type_display == 'Attention-LSTM':
            model_summary = st.session_state.attention_lstm_model.get_model_summary()
            st.code(model_summary, language='text')
        elif model_type_display == 'N-BEATS':
            model_summary = st.session_state.nbeats_model.get_model_summary()
            st.code(model_summary, language='text')
        elif model_type_display == 'TCN':
            model_summary = st.session_state.tcn_model.get_model_summary()
            st.code(model_summary, language='text')
        elif model_type_display == 'Transformer':
            model_summary = st.session_state.transformer_model.get_model_summary()
            st.code(model_summary, language='text')
        elif model_type_display == 'Ensemble':
            model_summary = st.session_state.ensemble_model.get_model_summary()
            st.code(model_summary, language='text')
        
        st.markdown("---")
        st.markdown("### 📈 Training History")
        
        # Check if training history exists (only available for models trained in current session)
        if 'history' in st.session_state and st.session_state.history is not None:
            with st.spinner("Generating training plots..."):
                # For LSTM, Attention-LSTM, TCN, Transformer: history is a Keras-like History object
                # For N-BEATS: history is a dict
                # For Ensemble: use first model's history
                if model_type_display in ['LSTM', 'Attention-LSTM', 'TCN', 'Transformer', 'Ensemble']:
                    fig = Visualizer.plot_training_history(st.session_state.history)
                else:  # N-BEATS
                    # Create a simple history object for visualization
                    class SimpleHistory:
                        def __init__(self, history_dict):
                            self.history = history_dict
                    
                    simple_hist = SimpleHistory({
                        'loss': st.session_state.history['train_loss'],
                        'val_loss': st.session_state.history['val_loss']
                    })
                    fig = Visualizer.plot_training_history(simple_hist)
                st.pyplot(fig)
            
            # Training metrics table
            st.markdown("#### Training Metrics Summary")
            if model_type_display in ['LSTM', 'Attention-LSTM', 'TCN', 'Transformer', 'Ensemble']:
                history_df = pd.DataFrame({
                    'Epoch': range(1, len(st.session_state.history.history['loss']) + 1),
                    'Training Loss': st.session_state.history.history['loss'],
                    'Validation Loss': st.session_state.history.history['val_loss'],
                })
            else:  # N-BEATS
                history_df = pd.DataFrame({
                    'Epoch': range(1, len(st.session_state.history['train_loss']) + 1),
                    'Training Loss': st.session_state.history['train_loss'],
                    'Validation Loss': st.session_state.history['val_loss'],
                })
            
            st.dataframe(
                history_df.tail(10).style.format({
                    'Training Loss': '{:.6f}',
                    'Validation Loss': '{:.6f}'
                }),
                width="stretch"
            )
        else:
            st.info("📝 **Training history is only available for models trained in the current session.**\n\nThe currently loaded model was trained previously, so historical training metrics are not available. You can retrain the model to view training progress.")
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Performance Metrics")
        
        if 'metrics' in st.session_state and st.session_state.metrics is not None:
            metrics = st.session_state.metrics
            
            # Display metrics in cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("RMSE", f"${metrics['rmse']:.4f}")
            
            with col2:
                st.metric("MAE", f"${metrics['mae']:.4f}")
            
            with col3:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
            with col4:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            with col5:
                st.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.2f}%")
        else:
            st.info("📝 **Performance metrics are only available for models trained in the current session.**\n\nTo evaluate the loaded model, navigate to the **Predictions** tab to see prediction results.")
        
        st.markdown("---")
        
        # Feature Importance
        if 'X_train' in st.session_state:
            st.markdown("### 📊 Feature Importance Analysis")
            with st.expander("View Feature Importance"):
                with st.spinner("Analyzing feature importance..."):
                    fig = Visualizer.plot_feature_importance_proxy(
                        st.session_state.X_train, 
                        config.FEATURES
                    )
                    st.pyplot(fig)
        
        st.markdown("---")
        
        # Metrics explanation
        with st.expander("📚 Metrics Explanation"):
            st.markdown("""
            **RMSE (Root Mean Squared Error):**
            - Measures the average magnitude of prediction errors
            - Lower values indicate better performance
            - Penalizes large errors more than small ones
            
            **MAE (Mean Absolute Error):**
            - Average absolute difference between predicted and actual values
            - Lower values indicate better accuracy
            - Less sensitive to outliers than RMSE
            
            **MAPE (Mean Absolute Percentage Error):**
            - Percentage-based error metric
            - Easier to interpret across different price ranges
            - Lower percentages indicate better predictions
            
            **R² Score (Coefficient of Determination):**
            - Measures how well predictions explain variance in actual data
            - Range:  -∞ to 1 (1 is perfect prediction)
            - Values closer to 1 indicate better fit
            
            **Direction Accuracy:**
            - Percentage of correct up/down movement predictions
            - Important for trading strategies
            - Higher percentages indicate better directional predictions
            """)
        
        # Performance interpretation
        if 'metrics' in st.session_state and st.session_state.metrics is not None:
            st.markdown("### 🎯 Performance Interpretation")
            
            metrics = st.session_state.metrics
            
            if metrics['mape'] < 5:
                st.success("🌟 **Excellent Performance:** MAPE < 5% - Very accurate predictions")
            elif metrics['mape'] < 10:
                st.success("✅ **Good Performance:** MAPE < 10% - Reliable predictions")
            elif metrics['mape'] < 15:
                st.info("📊 **Fair Performance:** MAPE < 15% - Acceptable predictions")
            else:
                st.warning("⚠️ **Poor Performance:** MAPE > 15% - Consider retraining or adjusting parameters")
            
            if metrics['direction_accuracy'] > 60:
                st.success(f"🎯 **Strong Directional Accuracy:** {metrics['direction_accuracy']:.1f}% - Good for trading signals")
            else:
                st.warning(f"📉 **Weak Directional Accuracy:** {metrics['direction_accuracy']:.1f}% - Not reliable for trading")
        
    else:
        st.info("👆 Click **'Train Model'** in the sidebar to start training.")
        
        # Show example of what the model will learn
        st.markdown("### 🧠 What Will the Model Learn?")
        st.markdown("""
        The LSTM model will analyze:
        1. **Historical Price Patterns:** Opening, closing, high, and low prices
        2. **Volume Trends:** Trading volume patterns
        3. **Technical Indicators:**
           - Moving Averages (50-day and 200-day)
           - RSI (Relative Strength Index)
           - MACD (Moving Average Convergence Divergence)
           - Bollinger Bands
        4. **Temporal Dependencies:** Relationships between past and future prices
        
        The model uses 60 days of historical data to predict the next day's price. 
        """)

# ==================== TAB 3: PREDICTIONS ====================
if active_tab == 3:
    st.markdown('<h2 class="sub-header">Model Predictions vs Actual Prices</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the **Model Training** tab.")
        st.stop()
    
    # Model selector for predictions
    st.markdown("### ⚙️ Prediction Settings")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Determine available models
        available_models = []
        if 'lstm_model' in st.session_state:
            available_models.append('LSTM')
        if 'attention_lstm_model' in st.session_state:
            available_models.append('Attention-LSTM')
        if 'nbeats_model' in st.session_state:
            available_models.append('N-BEATS')
        if 'tcn_model' in st.session_state:
            available_models.append('TCN')
        if 'transformer_model' in st.session_state:
            available_models.append('Transformer')
        if 'ensemble_model' in st.session_state:
            available_models.append('Ensemble')
        
        # Add "All Models" option if more than one model is available
        if len(available_models) > 1:
            model_options = ['All Models'] + available_models
        else:
            model_options = available_models
        
        prediction_model_choice = st.selectbox(
            "Select Model(s) for Prediction",
            options=model_options,
            help="Choose a single model or compare all trained models"
        )
    
    with col2:
        if prediction_model_choice == 'All Models':
            st.info(f"📊 Comparing {len(available_models)} models: {', '.join(available_models)}")
        else:
            st.info(f"🎯 Using {prediction_model_choice} for predictions")
    
    st.markdown("---")
    
    try:
        # Check if test data exists in session state, if not regenerate it
        if 'X_test' not in st.session_state or 'y_test' not in st.session_state:
            with st.spinner("📊 Preparing test data from loaded stock data..."):
                # Load data if not already loaded
                if not st.session_state.data_loaded:
                    df, stock_info, error = load_data(ticker, period)
                    if error:
                        st.error(f"❌ Error loading data: {error}")
                        st.stop()
                    df, error = add_features(df)
                    if error:
                        st.error(f"❌ Error adding features: {error}")
                        st.stop()
                    st.session_state.df = df
                    st.session_state.stock_info = stock_info
                    st.session_state.data_loaded = True
                
                df = st.session_state.df
                
                # Prepare data using the loaded preprocessor
                features = df[config.FEATURES].values
                target = df[[config.TARGET]].values
                
                # Normalize using loaded scalers (don't fit again!)
                normalized_features = st.session_state.preprocessor.feature_scaler.transform(features)
                normalized_target = st.session_state.preprocessor.target_scaler.transform(target)
                
                # Create sequences
                X, y = st.session_state.preprocessor.create_sequences(
                    normalized_features, 
                    normalized_target, 
                    seq_length=seq_length
                )
                
                # Split data
                split_idx = int(len(X) * (1 - config.TEST_SIZE))
                st.session_state.X_test = X[split_idx:]
                st.session_state.y_test = y[split_idx:]
                
                st.success("✅ Test data regenerated from current stock data")
        
        # Make predictions based on model selection
        y_test = st.session_state.y_test
        y_test_actual = st.session_state.preprocessor.inverse_transform_target(y_test)
        
        if prediction_model_choice == 'All Models':
            # Get predictions from all models
            model_predictions = {}
            
            with st.spinner("🔮 Making predictions with all models..."):
                for model_name in available_models:
                    if model_name == 'LSTM':
                        trainer = ModelTrainer(st.session_state.lstm_model)
                    elif model_name == 'Attention-LSTM':
                        trainer = ModelTrainer(st.session_state.attention_lstm_model)
                    elif model_name == 'N-BEATS':
                        trainer = NBeatsTrainer(st.session_state.nbeats_model)
                    elif model_name == 'TCN':
                        trainer = ModelTrainer(st.session_state.tcn_model)
                    elif model_name == 'Transformer':
                        trainer = ModelTrainer(st.session_state.transformer_model)
                    elif model_name == 'Ensemble':
                        # Ensemble has its own predict method
                        y_pred = st.session_state.ensemble_model.predict(st.session_state.X_test)
                        model_predictions[model_name] = st.session_state.preprocessor.inverse_transform_target(y_pred)
                        continue
                    
                    y_pred = trainer.predict(st.session_state.X_test)
                    model_predictions[model_name] = st.session_state.preprocessor.inverse_transform_target(y_pred)
            
            # Use first model's predictions as default for metrics
            y_pred_actual = model_predictions[available_models[0]]
            
        else:
            # Single model prediction
            with st.spinner(f"🔮 Making predictions with {prediction_model_choice}..."):
                if prediction_model_choice == 'LSTM':
                    trainer = ModelTrainer(st.session_state.lstm_model)
                elif prediction_model_choice == 'Attention-LSTM':
                    trainer = ModelTrainer(st.session_state.attention_lstm_model)
                elif prediction_model_choice == 'N-BEATS':
                    trainer = NBeatsTrainer(st.session_state.nbeats_model)
                elif prediction_model_choice == 'TCN':
                    trainer = ModelTrainer(st.session_state.tcn_model)
                elif prediction_model_choice == 'Transformer':
                    trainer = ModelTrainer(st.session_state.transformer_model)
                elif prediction_model_choice == 'Ensemble':
                    y_pred = st.session_state.ensemble_model.predict(st.session_state.X_test)
                    y_pred_actual = st.session_state.preprocessor.inverse_transform_target(y_pred)
                else:
                    trainer = st.session_state.trainer
                
                if prediction_model_choice != 'Ensemble':
                    y_pred = trainer.predict(st.session_state.X_test)
                    y_pred_actual = st.session_state.preprocessor.inverse_transform_target(y_pred)
            
            model_predictions = {prediction_model_choice: y_pred_actual}
        
        # Get dates for test set
        test_dates = st.session_state.df['Date'].iloc[-len(y_test):].reset_index(drop=True)
        
        st.success(f"✅ Generated {len(y_pred_actual)} predictions")
        
        # Summary statistics
        st.markdown("### 📊 Prediction Summary")
        
        errors = y_test_actual. flatten() - y_pred_actual.flatten()
        abs_errors = np.abs(errors)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Error", f"${np.mean(errors):.2f}")
        with col2:
            st.metric("Mean Absolute Error", f"${np.mean(abs_errors):.2f}")
        with col3:
            st.metric("Max Error", f"${np.max(abs_errors):.2f}")
        with col4:
            st.metric("Min Error", f"${np.min(abs_errors):.2f}")
        
        st.markdown("---")
        
        # Predictions vs Actual plot
        if prediction_model_choice == 'All Models':
            st.markdown("### 📈 Model Comparison: Predictions vs Actual")
            
            # Create comparison plot with all models
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=list(range(len(y_test_actual))),
                y=y_test_actual.flatten(),
                mode='lines',
                name='Actual',
                line=dict(color='#1f77b4', width=2.5)
            ))
            
            # Add predictions from each model
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            for i, (model_name, pred) in enumerate(model_predictions.items()):
                fig.add_trace(go.Scatter(
                    x=list(range(len(pred))),
                    y=pred.flatten(),
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=1.8, dash='dash')
                ))
            
            fig.update_layout(
                title='Model Comparison: Predictions vs Actual Prices',
                xaxis_title='Sample Index',
                yaxis_title='Stock Price ($)',
                height=500,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Show metrics comparison table
            st.markdown("### 📊 Model Performance Comparison")
            
            comparison_metrics = []
            for model_name, pred in model_predictions.items():
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                mse = mean_squared_error(y_test_actual, pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_actual, pred)
                r2 = r2_score(y_test_actual, pred)
                mape = np.mean(np.abs((y_test_actual - pred) / (y_test_actual + 1e-6))) * 100
                
                comparison_metrics.append({
                    'Model': model_name,
                    'RMSE': f"{rmse:.4f}",
                    'MAE': f"{mae:.4f}",
                    'MAPE (%)': f"{mape:.2f}",
                    'R² Score': f"{r2:.4f}"
                })
            
            metrics_df = pd.DataFrame(comparison_metrics)
            st.dataframe(metrics_df, width='stretch')
            
            # Highlight best model
            best_r2_idx = metrics_df['R² Score'].astype(float).idxmax()
            best_model = metrics_df.loc[best_r2_idx, 'Model']
            st.success(f"🏆 Best performing model: **{best_model}** (R² = {metrics_df.loc[best_r2_idx, 'R² Score']})")
            
            st.markdown("---")
            
            # Prediction Error Analysis for All Models - Combined
            st.markdown("### 📉 Prediction Error Analysis - All Models")
            with st.spinner("Analyzing errors for all models..."):
                fig = Visualizer.plot_multi_model_errors(y_test_actual, model_predictions)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Prediction with Confidence Intervals for All Models - Combined
            st.markdown("### 📊 Predictions with Confidence Intervals - All Models")
            st.info("💡 The shaded areas represent 95% confidence intervals for each model's predictions.")
            with st.spinner("Calculating confidence intervals for all models..."):
                fig = Visualizer.plot_multi_model_confidence_intervals(y_test_actual, model_predictions, confidence=0.95)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Comprehensive Residual Analysis for All Models - Combined
            st.markdown("### 🔬 Comprehensive Residual Analysis - All Models")
            st.markdown("""
            **📖 Interpretation Guide:**
            - **Top Left (Residuals vs Predicted):** Should show random scatter around zero with no patterns
            - **Top Right (Distribution):** Should be bell-shaped, indicating normally distributed errors
            - **Bottom Left (Q-Q Plot):** Points should follow the diagonal line, confirming normality
            - **Bottom Right (Residuals Over Time):** Should show random fluctuation around zero
            """)
            with st.spinner("Performing residual analysis for all models..."):
                fig = Visualizer.plot_multi_model_residuals(y_test_actual, model_predictions)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Detailed Prediction Comparison for All Models
            st.markdown("### 📋 Detailed Prediction Comparison - All Models")
            for model_name, y_pred_model in model_predictions.items():
                st.markdown(f"#### {model_name} - Prediction Details")
                
                # Calculate errors for this model
                model_errors = y_test_actual.flatten() - y_pred_model.flatten()
                model_abs_errors = np.abs(model_errors)
                
                model_comparison_df = pd.DataFrame({
                    'Date': test_dates,
                    'Actual Price': y_test_actual.flatten(),
                    'Predicted Price': y_pred_model.flatten(),
                    'Error': model_errors,
                    'Absolute Error': model_abs_errors,
                    'Error %': (model_errors / y_test_actual.flatten() * 100)
                })
                
                # Format display
                model_display_df = model_comparison_df.copy()
                model_display_df['Date'] = pd.to_datetime(model_display_df['Date']).dt.strftime('%Y-%m-%d')
                model_display_df['Actual Price'] = model_display_df['Actual Price'].apply(lambda x: f"${x:.2f}")
                model_display_df['Predicted Price'] = model_display_df['Predicted Price'].apply(lambda x: f"${x:.2f}")
                model_display_df['Error'] = model_display_df['Error'].apply(lambda x: f"${x:.2f}")
                model_display_df['Absolute Error'] = model_display_df['Absolute Error'].apply(lambda x: f"${x:.2f}")
                model_display_df['Error %'] = model_display_df['Error %'].apply(lambda x: f"{x:.2f}%")
                
                # Show recent predictions
                st.markdown("##### Most Recent Predictions (Last 20 Days)")
                st.dataframe(model_display_df.tail(20), width="stretch", hide_index=True)
                
                # Show best and worst predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 🎯 Best Predictions (Lowest Error)")
                    best_predictions = model_comparison_df.nsmallest(5, 'Absolute Error')[['Date', 'Actual Price', 'Predicted Price', 'Error %']]
                    best_predictions['Date'] = pd.to_datetime(best_predictions['Date']).dt.strftime('%Y-%m-%d')
                    best_predictions['Actual Price'] = best_predictions['Actual Price'].apply(lambda x: f"${x:.2f}")
                    best_predictions['Predicted Price'] = best_predictions['Predicted Price'].apply(lambda x: f"${x:.2f}")
                    best_predictions['Error %'] = best_predictions['Error %'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(best_predictions, width="stretch", hide_index=True)
                
                with col2:
                    st.markdown("##### ⚠️ Worst Predictions (Highest Error)")
                    worst_predictions = model_comparison_df.nlargest(5, 'Absolute Error')[['Date', 'Actual Price', 'Predicted Price', 'Error %']]
                    worst_predictions['Date'] = pd.to_datetime(worst_predictions['Date']).dt.strftime('%Y-%m-%d')
                    worst_predictions['Actual Price'] = worst_predictions['Actual Price'].apply(lambda x: f"${x:.2f}")
                    worst_predictions['Predicted Price'] = worst_predictions['Predicted Price'].apply(lambda x: f"${x:.2f}")
                    worst_predictions['Error %'] = worst_predictions['Error %'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(worst_predictions, width="stretch", hide_index=True)
                
                # Download predictions for this model
                csv = model_comparison_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {model_name} Predictions",
                    data=csv,
                    file_name=f"{ticker}_{model_name.lower().replace(' ', '_')}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key=f"download_{model_name}"
                )
                st.markdown("---")
            
        else:
            st.markdown("### 📈 Predictions vs Actual Prices")
            with st.spinner("Generating comparison plot..."):
                fig = Visualizer.plot_predictions(y_test_actual, y_pred_actual, test_dates)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Prediction errors
            st.markdown("### 📉 Prediction Error Analysis")
            with st.spinner("Analyzing errors..."):
                fig = Visualizer.plot_prediction_error(y_test_actual, y_pred_actual)
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Prediction with Confidence Intervals
            st.markdown("### 📊 Predictions with Confidence Intervals")
            with st.expander("View Confidence Analysis"):
                with st.spinner("Calculating confidence intervals..."):
                    fig = Visualizer.plot_prediction_intervals(y_test_actual, y_pred_actual, confidence=0.95)
                    st.pyplot(fig)
                    st.info("💡 The shaded area represents the 95% confidence interval. The model's predictions have a 95% probability of falling within this range.")
            
            st.markdown("---")
            
            # Comprehensive Residual Analysis
            st.markdown("### 🔬 Comprehensive Residual Analysis")
            with st.expander("View Detailed Residual Diagnostics"):
                with st.spinner("Performing residual analysis..."):
                    fig = Visualizer.plot_residual_analysis(y_test_actual, y_pred_actual)
                    st.pyplot(fig)
                    st.markdown("""
                    **📖 Interpretation Guide:**
                    - **Top Left (Residuals vs Predicted):** Should show random scatter around zero with no patterns
                    - **Top Right (Distribution):** Should be bell-shaped, indicating normally distributed errors
                    - **Bottom Left (Q-Q Plot):** Points should follow the diagonal line, confirming normality
                    - **Bottom Right (Residuals Over Time):** Should show random fluctuation around zero
                    """)
            
            st.markdown("---")
            
            # Detailed comparison table
            st.markdown("### 📋 Detailed Prediction Comparison")
            
            comparison_df = pd.DataFrame({
                'Date': test_dates,
                'Actual Price': y_test_actual.flatten(),
                'Predicted Price': y_pred_actual.flatten(),
                'Error': errors,
                'Absolute Error': abs_errors,
                'Error %': (errors / y_test_actual.flatten() * 100)
            })
            
            # Format display
            display_df = comparison_df.copy()
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
            display_df['Actual Price'] = display_df['Actual Price'].apply(lambda x: f"${x:.2f}")
            display_df['Predicted Price'] = display_df['Predicted Price'].apply(lambda x: f"${x:.2f}")
            display_df['Error'] = display_df['Error'].apply(lambda x: f"${x:.2f}")
            display_df['Absolute Error'] = display_df['Absolute Error'].apply(lambda x: f"${x:.2f}")
            display_df['Error %'] = display_df['Error %'].apply(lambda x: f"{x:.2f}%")
            
            # Show recent predictions
            st.markdown("#### Most Recent Predictions (Last 20 Days)")
            st.dataframe(display_df.tail(20), width="stretch", hide_index=True)
            
            # Show best and worst predictions
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Best Predictions (Lowest Error)")
                best_predictions = comparison_df.nsmallest(5, 'Absolute Error')[['Date', 'Actual Price', 'Predicted Price', 'Error %']]
                best_predictions['Date'] = pd.to_datetime(best_predictions['Date']).dt.strftime('%Y-%m-%d')
                best_predictions['Actual Price'] = best_predictions['Actual Price'].apply(lambda x: f"${x:.2f}")
                best_predictions['Predicted Price'] = best_predictions['Predicted Price'].apply(lambda x: f"${x:.2f}")
                best_predictions['Error %'] = best_predictions['Error %'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(best_predictions, width="stretch", hide_index=True)
            
            with col2:
                st.markdown("#### ⚠️ Worst Predictions (Highest Error)")
                worst_predictions = comparison_df.nlargest(5, 'Absolute Error')[['Date', 'Actual Price', 'Predicted Price', 'Error %']]
                worst_predictions['Date'] = pd.to_datetime(worst_predictions['Date']).dt.strftime('%Y-%m-%d')
                worst_predictions['Actual Price'] = worst_predictions['Actual Price'].apply(lambda x: f"${x:.2f}")
                worst_predictions['Predicted Price'] = worst_predictions['Predicted Price'].apply(lambda x: f"${x:.2f}")
                worst_predictions['Error %'] = worst_predictions['Error %'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(worst_predictions, width="stretch", hide_index=True)
            
            st.markdown("---")
            
            # Download predictions
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Predictions",
                data=csv,
                file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch"
            )
        
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")
        logger.error(f"Prediction error: {str(e)}", exc_info=True)

# ==================== TAB 4: FUTURE FORECAST ====================
if active_tab == 4:
    st.markdown('<h2 class="sub-header">Future Price Forecast</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the **Model Training** tab.")
        st.stop()
    
    # Model selector for forecasting
    st.markdown("### ⚙️ Forecast Settings")
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Determine available models
        available_models = []
        if 'lstm_model' in st.session_state:
            available_models.append('LSTM')
        if 'attention_lstm_model' in st.session_state:
            available_models.append('Attention-LSTM')
        if 'nbeats_model' in st.session_state:
            available_models.append('N-BEATS')
        if 'tcn_model' in st.session_state:
            available_models.append('TCN')
        if 'transformer_model' in st.session_state:
            available_models.append('Transformer')
        if 'ensemble_model' in st.session_state:
            available_models.append('Ensemble')
        
        # Add "All Models" option if more than one model is available
        if len(available_models) > 1:
            forecast_model_options = ['All Models'] + available_models
        else:
            forecast_model_options = available_models
        
        forecast_model_choice = st.selectbox(
            "Select Model(s) for Forecast",
            options=forecast_model_options,
            help="Choose a single model or compare forecasts from all trained models"
        )
    
    with col2:
        if forecast_model_choice == 'All Models':
            st.info(f"📊 Comparing forecasts from {len(available_models)} models")
        else:
            st.info(f"🎯 Using {forecast_model_choice} for forecasting")
    
    st.markdown("---")
    
    try:
        # Get last sequence for prediction
        features, target, _ = st.session_state.preprocessor.normalize_data(st.session_state.df)
        X, y = st.session_state.preprocessor.create_sequences(features, target, seq_length)
        last_sequence = X[-1]
        
        # Make future predictions based on model selection
        if forecast_model_choice == 'All Models':
            model_forecasts = {}
            
            with st.spinner(f"🔮 Predicting next {future_days} days with all models..."):
                for model_name in available_models:
                    if model_name == 'LSTM':
                        trainer = ModelTrainer(st.session_state.lstm_model)
                    elif model_name == 'Attention-LSTM':
                        trainer = ModelTrainer(st.session_state.attention_lstm_model)
                    elif model_name == 'N-BEATS':
                        trainer = NBeatsTrainer(st.session_state.nbeats_model)
                    elif model_name == 'TCN':
                        trainer = ModelTrainer(st.session_state.tcn_model)
                    elif model_name == 'Transformer':
                        trainer = ModelTrainer(st.session_state.transformer_model)
                    elif model_name == 'Ensemble':
                        # Ensemble needs custom logic
                        predictions = []
                        current_seq = last_sequence.copy()
                        for _ in range(future_days):
                            pred = st.session_state.ensemble_model.predict(current_seq.reshape(1, seq_length, -1))
                            predictions.append(pred[0])
                            # Update sequence
                            current_seq = np.roll(current_seq, -1, axis=0)
                            current_seq[-1] = np.concatenate([pred, current_seq[-1, 1:]])
                        model_forecasts[model_name] = st.session_state.preprocessor.inverse_transform_target(np.array(predictions))
                        continue
                    
                    forecast = trainer.predict_future(last_sequence, future_days, st.session_state.preprocessor)
                    model_forecasts[model_name] = forecast
            
            # Use first model's forecast as default for metrics
            future_predictions = model_forecasts[available_models[0]]
            
        else:
            # Single model forecast
            with st.spinner(f"🔮 Predicting next {future_days} days with {forecast_model_choice}..."):
                if forecast_model_choice == 'LSTM':
                    trainer = ModelTrainer(st.session_state.lstm_model)
                elif forecast_model_choice == 'Attention-LSTM':
                    trainer = ModelTrainer(st.session_state.attention_lstm_model)
                elif forecast_model_choice == 'N-BEATS':
                    trainer = NBeatsTrainer(st.session_state.nbeats_model)
                elif forecast_model_choice == 'TCN':
                    trainer = ModelTrainer(st.session_state.tcn_model)
                elif forecast_model_choice == 'Transformer':
                    trainer = ModelTrainer(st.session_state.transformer_model)
                elif forecast_model_choice == 'Ensemble':
                    predictions = []
                    current_seq = last_sequence.copy()
                    for _ in range(future_days):
                        pred = st.session_state.ensemble_model.predict(current_seq.reshape(1, seq_length, -1))
                        predictions.append(pred[0])
                        current_seq = np.roll(current_seq, -1, axis=0)
                        current_seq[-1] = np.concatenate([pred, current_seq[-1, 1:]])
                    future_predictions = st.session_state.preprocessor.inverse_transform_target(np.array(predictions))
                else:
                    trainer = st.session_state.trainer
                
                if forecast_model_choice != 'Ensemble':
                    future_predictions = trainer.predict_future(last_sequence, future_days, st.session_state.preprocessor)
            
            model_forecasts = {forecast_model_choice: future_predictions}
        
        st.success(f"✅ Generated {future_days}-day forecast")
        
        # Create future dates
        last_date = st.session_state.df['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=future_days,
            freq='D'
        )
        
        # Current price and statistics
        current_price = st. session_state.df['Close']. iloc[-1]
        predicted_price_end = future_predictions[-1][0]
        price_change = predicted_price_end - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Summary metrics
        st.markdown("### 📊 Forecast Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric(
                f"Predicted ({future_days}d)",
                f"${predicted_price_end:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        
        with col3:
            st.metric("Price Change", f"${price_change:.2f}")
        
        with col4:
            predicted_high = future_predictions. max()
            st.metric("Predicted High", f"${predicted_high:.2f}")
        
        with col5:
            predicted_low = future_predictions.min()
            st.metric("Predicted Low", f"${predicted_low:.2f}")
        
        # Trend indicator
        if price_change_pct > 5:
            st.success("📈 **Strong Bullish Trend** - Significant upward movement predicted")
        elif price_change_pct > 0:
            st.info("📈 **Bullish Trend** - Upward movement predicted")
        elif price_change_pct > -5:
            st.info("📉 **Bearish Trend** - Downward movement predicted")
        else:
            st.error("📉 **Strong Bearish Trend** - Significant downward movement predicted")
        
        st.markdown("---")
        
        # Short, medium, long term forecasts
        st.markdown("### 📅 Multi-Period Forecast")
        
        col1, col2, col3 = st.columns(3)
        
        # 7-day forecast
        if future_days >= 7:
            with col1:
                pred_7d = future_predictions[6][0]
                change_7d = pred_7d - current_price
                change_7d_pct = (change_7d / current_price) * 100
                
                st.markdown("#### 7-Day Forecast")
                st.metric(
                    "Predicted Price",
                    f"${pred_7d:.2f}",
                    delta=f"{change_7d_pct:+.2f}%"
                )
                st.write(f"Change:  ${change_7d:+.2f}")
        
        # 14-day forecast
        if future_days >= 14:
            with col2:
                pred_14d = future_predictions[13][0]
                change_14d = pred_14d - current_price
                change_14d_pct = (change_14d / current_price) * 100
                
                st.markdown("#### 14-Day Forecast")
                st.metric(
                    "Predicted Price",
                    f"${pred_14d:.2f}",
                    delta=f"{change_14d_pct:+.2f}%"
                )
                st.write(f"Change: ${change_14d:+.2f}")
        
        # 30-day forecast
        if future_days >= 30:
            with col3:
                pred_30d = future_predictions[29][0]
                change_30d = pred_30d - current_price
                change_30d_pct = (change_30d / current_price) * 100
                
                st. markdown("#### 30-Day Forecast")
                st.metric(
                    "Predicted Price",
                    f"${pred_30d:.2f}",
                    delta=f"{change_30d_pct:+.2f}%"
                )
                st.write(f"Change: ${change_30d:+.2f}")
        
        st.markdown("---")
        
        # Forecast visualization
        if forecast_model_choice == 'All Models':
            st.markdown("### 📈 Multi-Model Forecast Comparison")
            
            # Create comparison plot with all models
            fig = go.Figure()
            
            # Historical data
            historical_prices = st.session_state.df['Close'].tail(60).values
            historical_dates = st.session_state.df['Date'].tail(60).values
            
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_prices,
                mode='lines',
                name='Historical',
                line=dict(color='#1f77b4', width=2.5)
            ))
            
            # Forecast from each model
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            for i, (model_name, forecast) in enumerate(model_forecasts.items()):
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast.flatten(),
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title=f'{future_days}-Day Price Forecast Comparison',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                height=550,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Forecast comparison statistics
            st.markdown("### 📊 Forecast Statistics Comparison")
            
            forecast_stats = []
            for model_name, forecast in model_forecasts.items():
                predicted_end = forecast[-1][0]
                price_change = predicted_end - current_price
                price_change_pct = (price_change / current_price) * 100
                
                forecast_stats.append({
                    'Model': model_name,
                    f'Day {future_days} Price': f"${predicted_end:.2f}",
                    'Change ($)': f"${price_change:+.2f}",
                    'Change (%)': f"{price_change_pct:+.2f}%",
                    'Predicted High': f"${forecast.max():.2f}",
                    'Predicted Low': f"${forecast.min():.2f}",
                    'Avg Daily Change': f"${np.mean(np.diff(forecast.flatten())):.2f}"
                })
            
            stats_df = pd.DataFrame(forecast_stats)
            st.dataframe(stats_df, width='stretch')
            
            # Show consensus/divergence
            st.markdown("### 🎯 Forecast Consensus")
            
            all_final_prices = [f[-1][0] for f in model_forecasts.values()]
            avg_forecast = np.mean(all_final_prices)
            std_forecast = np.std(all_final_prices)
            min_forecast = np.min(all_final_prices)
            max_forecast = np.max(all_final_prices)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Consensus Forecast", f"${avg_forecast:.2f}")
            with col2:
                consensus_change = ((avg_forecast - current_price) / current_price) * 100
                st.metric("Consensus Change", f"{consensus_change:+.2f}%")
            with col3:
                st.metric("Forecast Range", f"${max_forecast - min_forecast:.2f}")
            with col4:
                st.metric("Std Deviation", f"${std_forecast:.2f}")
            
            # Consensus interpretation
            divergence_pct = (std_forecast / current_price) * 100
            if divergence_pct < 2:
                st.success("✅ **High Consensus**: Models agree closely on the forecast (divergence < 2%)")
            elif divergence_pct < 5:
                st.info("ℹ️ **Moderate Consensus**: Some variation in model forecasts (divergence 2-5%)")
            else:
                st.warning("⚠️ **Low Consensus**: Significant divergence between model forecasts (divergence > 5%)")
            
            # Show which models are bullish/bearish
            bullish_models = [name for name, f in model_forecasts.items() if f[-1][0] > current_price]
            bearish_models = [name for name, f in model_forecasts.items() if f[-1][0] <= current_price]
            
            col1, col2 = st.columns(2)
            with col1:
                if bullish_models:
                    st.success(f"📈 **Bullish Models ({len(bullish_models)})**: {', '.join(bullish_models)}")
            with col2:
                if bearish_models:
                    st.error(f"📉 **Bearish Models ({len(bearish_models)})**: {', '.join(bearish_models)}")
            
        else:
            st.markdown("### 📈 Future Price Forecast Chart")
            
            with st.spinner("Generating forecast chart..."):
                historical_prices = st.session_state.df['Close'].tail(90)
                historical_dates = st.session_state.df['Date'].tail(90)
                
                fig = Visualizer.plot_future_predictions(
                    historical_prices,
                    future_predictions.flatten(),
                    historical_dates,
                    future_dates,
                    ticker
                )
                st.pyplot(fig)
        
        st. markdown("---")
        
        # Detailed forecast table
        st.markdown("### 📋 Detailed Forecast Table")
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_predictions.flatten(),
            'Change from Current': future_predictions.flatten() - current_price,
            'Change %': ((future_predictions.flatten() - current_price) / current_price * 100)
        })
        
        # Format display
        display_forecast = forecast_df.copy()
        display_forecast['Date'] = pd.to_datetime(display_forecast['Date']).dt.strftime('%Y-%m-%d')
        display_forecast['Predicted Price'] = display_forecast['Predicted Price'].apply(lambda x: f"${x:.2f}")
        display_forecast['Change from Current'] = display_forecast['Change from Current'].apply(lambda x: f"${x:+.2f}")
        display_forecast['Change %'] = display_forecast['Change %'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(display_forecast, width="stretch", hide_index=True)
        
        st.markdown("---")
        
        # Investment scenario
        st.markdown("### 💰 Investment Scenario Calculator")
        
        col1, col2 = st. columns(2)
        
        with col1:
            investment_amount = st.number_input(
                "Investment Amount ($)",
                min_value=100,
                max_value=1000000,
                value=10000,
                step=100
            )
        
        with col2:
            shares = investment_amount / current_price
            predicted_value = shares * predicted_price_end
            potential_gain = predicted_value - investment_amount
            roi = (potential_gain / investment_amount) * 100
            
            st.metric("Shares Purchased", f"{shares:.2f}")
            st.metric("Predicted Value", f"${predicted_value:.2f}")
            st.metric("Potential Gain/Loss", f"${potential_gain:+.2f}", delta=f"{roi:+.2f}%")
        
        st.markdown("---")
        
        # Download forecast
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast",
            data=csv,
            file_name=f"{ticker}_forecast_{datetime. now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width="stretch"
        )
        
        # Disclaimer
        st.markdown("---")
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("""
        ⚠️ **IMPORTANT DISCLAIMER**
        
        This forecast is generated by a machine learning model and is for **educational purposes only**. 
        
        **Key Limitations:**
        - Based solely on historical price patterns
        - Does not account for future news, earnings, or market events
        - Prediction accuracy decreases for longer time horizons
        - Market conditions can change rapidly
        
        **DO NOT use this as the sole basis for investment decisions.**
        Always consult with qualified financial advisors and conduct thorough research before investing.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e: 
        st.error(f"❌ Error generating forecast: {str(e)}")
        logger.error(f"Forecast error: {str(e)}", exc_info=True)

# ==================== TAB 5: REPORTS ====================
if active_tab == 11:
    st.markdown('<h2 class="sub-header">Comprehensive Analysis Report</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first to generate reports.")
        st.stop()
    
    # Generate comprehensive report
    st.markdown("### 📄 Executive Summary")
    
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    st.markdown(f"""
    **Report Generated:** {report_date}  
    **Stock Ticker:** {ticker}  
    **Company:** {st.session_state.stock_info. get('name', 'N/A')}  
    **Data Period:** {period_label}  
    **Model Type:** LSTM Neural Network
    """)
    
    st.markdown("---")
    
    # Model Performance Summary
    st.markdown("### 🎯 Model Performance Summary")
    
    if st.session_state.metrics is not None:
        metrics = st.session_state.metrics
        
        performance_data = {
            'Metric':  ['RMSE', 'MAE', 'MAPE', 'R² Score', 'Direction Accuracy'],
            'Value': [
                f"${metrics['rmse']:.4f}",
                f"${metrics['mae']:.4f}",
                f"{metrics['mape']:.2f}%",
                f"{metrics['r2']:.4f}",
                f"{metrics['direction_accuracy']:.2f}%"
            ],
            'Interpretation': [
                'Lower is better' if metrics['rmse'] < 10 else 'Needs improvement',
                'Lower is better' if metrics['mae'] < 5 else 'Needs improvement',
                'Excellent' if metrics['mape'] < 5 else 'Good' if metrics['mape'] < 10 else 'Fair',
                'Excellent' if metrics['r2'] > 0.9 else 'Good' if metrics['r2'] > 0.8 else 'Fair',
                'Strong' if metrics['direction_accuracy'] > 60 else 'Weak'
            ]
        }
        
        st.dataframe(pd.DataFrame(performance_data), width="stretch", hide_index=True)
    else:
        st.info("📝 **Performance metrics are only available for models trained in the current session.**\n\nTo evaluate the loaded model, navigate to the **Predictions** tab to generate metrics.")
    
    st.markdown("---")
    
    # Technical Analysis Summary
    st.markdown("### 📊 Technical Analysis Summary")
    
    df = st.session_state.df
    current_price = df['Close'].iloc[-1]
    ma_50 = df['MA_50'].iloc[-1]
    ma_200 = df['MA_200'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    technical_summary = {
        'Indicator': ['Current Price', 'MA(50)', 'MA(200)', 'RSI', 'MACD Signal'],
        'Value': [
            f"${current_price:.2f}",
            f"${ma_50:.2f}",
            f"${ma_200:.2f}",
            f"{rsi:.2f}",
            'Bullish' if df['MACD']. iloc[-1] > df['MACD_Signal'].iloc[-1] else 'Bearish'
        ],
        'Signal': [
            '-',
            'Bullish' if current_price > ma_50 else 'Bearish',
            'Bullish' if ma_50 > ma_200 else 'Bearish',
            'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral',
            'Bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'Bearish'
        ]
    }
    
    st.dataframe(pd.DataFrame(technical_summary), width="stretch", hide_index=True)
    
    st.markdown("---")
    
    # Key Findings
    st.markdown("### 🔑 Key Findings")
    
    findings = []
    
    # Price trend
    if ma_50 > ma_200:
        findings.append("✅ **Golden Cross:** MA(50) above MA(200) indicates long-term bullish trend")
    else:
        findings.append("⚠️ **Death Cross:** MA(50) below MA(200) indicates long-term bearish trend")
    
    # RSI
    if rsi > 70:
        findings.append(f"⚠️ **Overbought:** RSI at {rsi:.1f} suggests potential price correction")
    elif rsi < 30:
        findings.append(f"✅ **Oversold:** RSI at {rsi:.1f} suggests potential price increase")
    else:
        findings.append(f"📊 **Neutral:** RSI at {rsi:.1f} is in normal range")
    
    # Model accuracy
    if metrics['direction_accuracy'] > 60:
        findings.append(f"✅ **Strong Predictions:** {metrics['direction_accuracy']:.1f}% direction accuracy")
    else:
        findings.append(f"⚠️ **Weak Predictions:** {metrics['direction_accuracy']:.1f}% direction accuracy")
    
    # Error rate
    if metrics['mape'] < 5:
        findings.append(f"🌟 **Excellent Accuracy:** {metrics['mape']:.2f}% average error")
    elif metrics['mape'] < 10:
        findings.append(f"✅ **Good Accuracy:** {metrics['mape']:.2f}% average error")
    else:
        findings.append(f"⚠️ **Fair Accuracy:** {metrics['mape']:.2f}% average error")
    
    for finding in findings:
        st. markdown(finding)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    recommendations = []
    
    if ma_50 > ma_200 and rsi < 70:
        recommendations.append("🟢 **Potential Buy Signal:** Bullish trend with room for growth")
    elif ma_50 < ma_200 and rsi > 30:
        recommendations.append("🔴 **Potential Sell Signal:** Bearish trend with downside risk")
    else:
        recommendations.append("🟡 **Hold Position:** Mixed signals suggest waiting for clearer trend")
    
    if metrics['direction_accuracy'] < 55:
        recommendations.append("⚠️ **Model Retraining Recommended:** Low directional accuracy")
    
    if metrics['mape'] > 10:
        recommendations.append("⚠️ **Use Caution:** High error rate suggests unreliable predictions")
    
    recommendations.append("📚 **Always Diversify:** Don't rely on single stock predictions")
    recommendations.append("🔍 **Monitor News:** Check for company announcements and market events")
    recommendations.append("👥 **Consult Professionals:** Seek advice from qualified financial advisors")
    
    for rec in recommendations:
        st. markdown(rec)
    
    st.markdown("---")
    
    # Export full report
    st.markdown("### 📥 Export Report")
    
    # Create comprehensive report text
    report_text = f"""
STOCK PRICE PREDICTION REPORT
{'='*80}

Report Generated:  {report_date}
Stock Ticker: {ticker}
Company: {st.session_state.stock_info.get('name', 'N/A')}
Sector: {st.session_state.stock_info.get('sector', 'N/A')}
Data Period: {period_label}

MODEL PERFORMANCE METRICS
{'-'*80}
RMSE (Root Mean Squared Error): ${metrics['rmse']:.4f}
MAE (Mean Absolute Error): ${metrics['mae']:.4f}
MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%
R² Score: {metrics['r2']:.4f}
Direction Accuracy: {metrics['direction_accuracy']:.2f}%

TECHNICAL ANALYSIS
{'-'*80}
Current Price: ${current_price:.2f}
MA(50): ${ma_50:.2f}
MA(200): ${ma_200:.2f}
RSI:  {rsi:.2f}
MACD Signal: {'Bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'Bearish'}

KEY FINDINGS
{'-'*80}
"""
    # Add findings
    for finding in findings:
        clean_finding = finding.replace('✅', '').replace('⚠️', '').replace('📊', '').replace('🌟', '').strip()
        report_text += f"- {clean_finding}\n"
    
    report_text += f"""
RECOMMENDATIONS
{'-'*80}
"""
    # Add recommendations
    for rec in recommendations:
        clean_rec = rec.replace('🟢', '').replace('🔴', '').replace('🟡', '').replace('⚠️', '').replace('📚', '').replace('🔍', '').replace('👥', '').strip()
        report_text += f"- {clean_rec}\n"
    
    report_text += f"""
DISCLAIMER
{'-'*80}
This report is generated by an AI-powered machine learning model and is for 
educational purposes only. Stock price predictions are inherently uncertain 
and should NOT be used as the sole basis for investment decisions.  Past 
performance does not guarantee future results.  Always consult with qualified 
financial advisors before making investment decisions. 

{'='*80}
"""
    
    col1, col2 = st. columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download Text Report",
            data=report_text,
            file_name=f"{ticker}_report_{datetime. now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            width="stretch"
        )
    
    with col2:
        # Create CSV summary
        summary_csv = pd.DataFrame({
            'Metric': ['Ticker', 'Date', 'Current Price', 'RMSE', 'MAE', 'MAPE', 'R²', 'Direction Accuracy', 'RSI', 'MA(50)', 'MA(200)'],
            'Value': [ticker, report_date, f"${current_price:.2f}", f"${metrics['rmse']:.4f}", 
                     f"${metrics['mae']:.4f}", f"{metrics['mape']:.2f}%", f"{metrics['r2']:.4f}",
                     f"{metrics['direction_accuracy']:.2f}%", f"{rsi:.2f}", f"${ma_50:.2f}", f"${ma_200:.2f}"]
        }).to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV Summary",
            data=summary_csv,
            file_name=f"{ticker}_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            width="stretch"
        )

# Tab 6: Trading Signals
if active_tab == 5:
    st.markdown('<h2 class="sub-header">💡 Trading Signals & Recommendations</h2>', unsafe_allow_html=True)
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Generate signals
        with st.spinner("🔍 Analyzing indicators and generating signals..."):
            signals_data = TradingSignals.get_comprehensive_signals(df)
        
        # Display overall recommendation
        overall = signals_data['overall']
        signal_emoji = TradingSignals.get_signal_emoji(overall['recommendation'])
        signal_color = TradingSignals.get_signal_color(overall['recommendation'])
        
        st.markdown(f"""
        <div style='
            padding: 2rem; 
            border-radius: 20px; 
            background: linear-gradient(135deg, {signal_color}22 0%, {signal_color}44 100%);
            border: 3px solid {signal_color};
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        '>
            <h1 style='font-size: 3rem; margin: 0;'>{signal_emoji}</h1>
            <h2 style='color: {signal_color}; margin: 1rem 0;'>{overall['recommendation']}</h2>
            <p style='font-size: 1.3rem; margin: 0; color: #1e293b;'>
                Confidence: <strong>{overall['confidence']:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📈 Buy Score", f"{overall['buy_score']:.1f}%", 
                     help="Aggregated bullish signal strength")
        
        with col2:
            st.metric("📉 Sell Score", f"{overall['sell_score']:.1f}%",
                     help="Aggregated bearish signal strength")
        
        st.markdown("---")
        
        # Individual signals
        st.markdown("### 📊 Indicator Breakdown")
        
        for indicator, signal in signals_data['signals'].items():
            with st.expander(f"{'🟢' if signal['signal'] == 'BUY' else '🔴' if signal['signal'] == 'SELL' else '⚪'} {indicator} - {signal['signal']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Signal:** {signal['signal']}")
                
                with col2:
                    st.markdown(f"**Strength:** {signal['strength']}")
                
                with col3:
                    st.markdown(f"**Indicator:** {signal['indicator']}")
                
                st.info(f"💡 {signal['reason']}")
        
        # Price and indicator summary
        st.markdown("---")
        st.markdown("### 📈 Current Market Data")
        
        latest = df.iloc[-1]
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"${latest['Close']:.2f}")
        
        with col2:
            if 'RSI' in df.columns:
                st.metric("RSI", f"{latest['RSI']:.1f}")
        
        with col3:
            if 'MA_50' in df.columns:
                st.metric("MA(50)", f"${latest['MA_50']:.2f}")
        
        with col4:
            if 'MA_200' in df.columns:
                st.metric("MA(200)", f"${latest['MA_200']:.2f}")
        
        with col5:
            if 'MACD' in df.columns:
                macd_signal = "Bullish" if latest['MACD'] > latest['MACD_Signal'] else "Bearish"
                st.metric("MACD", macd_signal)
        
        # Disclaimer
        st.warning("⚠️ **Disclaimer:** These signals are generated algorithmically and should not be considered financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.")
    
    else:
        st.info("📊 Please load stock data from the 'Data & Analysis' tab to view trading signals.")

# Tab 7: Stock Comparison
if active_tab == 6:
    st.markdown('<h2 class="sub-header">📊 Multi-Stock Comparison</h2>', unsafe_allow_html=True)
    
    # Stock selection for comparison
    st.markdown("### Select Stocks to Compare")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker1 = st.text_input("Stock 1", value=ticker, key="compare_ticker1").upper()
    
    with col2:
        ticker2 = st.text_input("Stock 2", value="MSFT", key="compare_ticker2").upper()
    
    with col3:
        ticker3 = st.text_input("Stock 3 (Optional)", value="", key="compare_ticker3").upper()
    
    comparison_period = st.selectbox(
        "Comparison Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=3,
        key="comparison_period"
    )
    
    if st.button("🔍 Compare Stocks", type="primary"):
        tickers_to_compare = [t for t in [ticker1, ticker2, ticker3] if t]
        
        if len(tickers_to_compare) < 2:
            st.error("Please enter at least 2 stock tickers to compare.")
        else:
            fetcher = StockDataFetcher()
            comparison_data = {}
            
            # Fetch data for all tickers
            progress_bar = st.progress(0)
            for idx, tick in enumerate(tickers_to_compare):
                with st.spinner(f"Fetching data for {tick}..."):
                    try:
                        stock_df = fetcher.fetch_stock_data(tick, period=comparison_period)
                        if stock_df is not None and not stock_df.empty:
                            comparison_data[tick] = stock_df
                        progress_bar.progress((idx + 1) / len(tickers_to_compare))
                    except Exception as e:
                        st.error(f"Error fetching {tick}: {str(e)}")
            
            if comparison_data:
                # Price comparison chart
                st.markdown("### 📈 Price Comparison (Normalized)")
                
                fig = go.Figure()
                
                for tick, data in comparison_data.items():
                    # Normalize to starting price
                    normalized = (data['Close'] / data['Close'].iloc[0]) * 100
                    fig.add_trace(go.Scatter(
                        x=data['Date'],
                        y=normalized,
                        mode='lines',
                        name=tick,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="Normalized Price Performance (Base 100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Metrics comparison
                st.markdown("### 📊 Key Metrics Comparison")
                
                metrics_data = []
                for tick, data in comparison_data.items():
                    # Add technical indicators
                    data = FeatureEngineer.add_all_indicators(data)
                    
                    latest = data.iloc[-1]
                    first = data.iloc[0]
                    
                    # Calculate returns
                    total_return = ((latest['Close'] - first['Close']) / first['Close']) * 100
                    avg_volume = data['Volume'].mean()
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
                    
                    metrics_data.append({
                        'Ticker': tick,
                        'Current Price': f"${latest['Close']:.2f}",
                        'Return %': f"{total_return:+.2f}%",
                        'Volatility': f"{volatility:.2f}%",
                        'Avg Volume': f"{avg_volume:,.0f}",
                        'RSI': f"{latest.get('RSI', 0):.1f}",
                        'MA(50)': f"${latest.get('MA_50', 0):.2f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, width='stretch', hide_index=True)
                
                # Volume comparison
                st.markdown("### 📊 Volume Comparison")
                
                fig_volume = go.Figure()
                
                for tick, data in comparison_data.items():
                    fig_volume.add_trace(go.Bar(
                        x=data['Date'],
                        y=data['Volume'],
                        name=tick,
                        opacity=0.7
                    ))
                
                fig_volume.update_layout(
                    title="Trading Volume Comparison",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_volume, width='stretch')
                
                # Export comparison
                st.markdown("### 💾 Export Comparison Data")
                
                # Combine all data
                export_data = pd.DataFrame()
                for tick, data in comparison_data.items():
                    temp_df = data[['Date', 'Close', 'Volume']].copy()
                    temp_df.columns = ['Date', f'{tick}_Price', f'{tick}_Volume']
                    if export_data.empty:
                        export_data = temp_df
                    else:
                        export_data = export_data.merge(temp_df, on='Date', how='outer')
                
                # Remove timezone from Date column for Excel compatibility
                if 'Date' in export_data.columns:
                    if pd.api.types.is_datetime64_any_dtype(export_data['Date']):
                        export_data['Date'] = export_data['Date'].dt.tz_localize(None)
                
                # Create Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    export_data.to_excel(writer, sheet_name='Price Data', index=False)
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                st.download_button(
                    label="📥 Download Comparison (Excel)",
                    data=output.getvalue(),
                    file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Tab 8: Portfolio Tracker
if active_tab == 7:
    st.markdown('<h2 class="sub-header">💼 Virtual Portfolio Tracker</h2>', unsafe_allow_html=True)
    
    portfolio = st.session_state.portfolio
    
    # Portfolio summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💰 Portfolio Overview")
    
    with col2:
        if st.button("🔄 Reset Portfolio"):
            st.session_state.portfolio = PortfolioManager(initial_cash=100000)
            st.success("Portfolio reset successfully!")
            st.rerun()
    
    # Get current prices for all holdings
    fetcher = StockDataFetcher()
    current_prices = {'cash': portfolio.cash}
    
    if portfolio.holdings:
        for tick in portfolio.holdings.keys():
            try:
                stock_data = fetcher.fetch_stock_data(tick, period='1d')
                if stock_data is not None and not stock_data.empty:
                    current_prices[tick] = stock_data['Close'].iloc[-1]
                else:
                    current_prices[tick] = portfolio.holdings[tick]['avg_price']
            except:
                current_prices[tick] = portfolio.holdings[tick]['avg_price']
    
    # Get portfolio value and metrics
    portfolio_value = portfolio.get_portfolio_value(current_prices)
    metrics = portfolio.get_performance_metrics(current_prices)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"${portfolio_value['total_value']:,.2f}",
            f"{portfolio_value['total_return_pct']:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${portfolio_value['cash']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Holdings Value",
            f"${portfolio_value['holdings_value']:,.2f}"
        )
    
    with col4:
        st.metric(
            "Total Return",
            f"${portfolio_value['total_return']:+,.2f}"
        )
    
    st.markdown("---")
    
    # Buy/Sell Interface
    st.markdown("### 💱 Trade Stocks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Buy Stock")
        
        buy_ticker = st.text_input("Ticker to Buy", value=ticker, key="buy_ticker").upper()
        buy_shares = st.number_input("Number of Shares", min_value=1, value=10, step=1, key="buy_shares")
        
        # Get current price
        if buy_ticker:
            try:
                stock_data = fetcher.fetch_stock_data(buy_ticker, period='1d')
                if stock_data is not None and not stock_data.empty:
                    buy_price = stock_data['Close'].iloc[-1]
                    st.info(f"Current Price: ${buy_price:.2f}")
                    st.info(f"Total Cost: ${buy_price * buy_shares:,.2f} (+ commission)")
                    
                    if st.button("🛒 Buy Stock", type="primary"):
                        result = portfolio.buy_stock(buy_ticker, buy_shares, buy_price)
                        if result['success']:
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error(result['message'])
                else:
                    st.error("Could not fetch current price")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("#### 🔴 Sell Stock")
        
        if portfolio.holdings:
            sell_ticker = st.selectbox(
                "Ticker to Sell",
                options=list(portfolio.holdings.keys()),
                key="sell_ticker"
            )
            
            max_shares = portfolio.holdings[sell_ticker]['shares']
            sell_shares = st.number_input(
                f"Shares to Sell (Max: {max_shares})",
                min_value=1,
                max_value=max_shares,
                value=min(10, max_shares),
                step=1,
                key="sell_shares"
            )
            
            # Get current price
            try:
                stock_data = fetcher.fetch_stock_data(sell_ticker, period='1d')
                if stock_data is not None and not stock_data.empty:
                    sell_price = stock_data['Close'].iloc[-1]
                    st.info(f"Current Price: ${sell_price:.2f}")
                    st.info(f"Total Proceeds: ${sell_price * sell_shares:,.2f} (- commission)")
                    
                    if st.button("💰 Sell Stock", type="primary"):
                        result = portfolio.sell_stock(sell_ticker, sell_shares, sell_price)
                        if result['success']:
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error(result['message'])
                else:
                    st.error("Could not fetch current price")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("No holdings to sell. Buy some stocks first!")
    
    st.markdown("---")
    
    # Current Holdings
    st.markdown("### 📋 Current Holdings")
    
    if portfolio.holdings:
        holdings_df = portfolio.get_holdings_df(current_prices)
        
        # Style the dataframe
        st.dataframe(
            holdings_df.style.format({
                'Avg Price': '${:.2f}',
                'Current Price': '${:.2f}',
                'Value': '${:,.2f}',
                'Cost Basis': '${:,.2f}',
                'P/L $': '${:+,.2f}',
                'P/L %': '{:+.2f}%'
            }),
            width='stretch',
            hide_index=True
        )
    else:
        st.info("No current holdings. Start buying stocks to build your portfolio!")
    
    st.markdown("---")
    
    # Transaction History
    st.markdown("### 📜 Transaction History")
    
    transactions_df = portfolio.get_transactions_df()
    
    if not transactions_df.empty:
        # Display recent transactions
        st.dataframe(
            transactions_df.sort_values('date', ascending=False).head(20),
            width='stretch',
            hide_index=True
        )
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        
        with col2:
            st.metric("Wins / Losses", f"{metrics['wins']} / {metrics['losses']}")
        
        with col3:
            if metrics['avg_win'] > 0:
                st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
            else:
                st.metric("Avg Win", "N/A")
        
        with col4:
            if metrics['avg_loss'] < 0:
                st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
            else:
                st.metric("Avg Loss", "N/A")
        
        # Export portfolio data
        st.markdown("### 💾 Export Portfolio Data")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if not holdings_df.empty:
                holdings_df.to_excel(writer, sheet_name='Holdings', index=False)
            
            # Remove timezone from date column for Excel compatibility
            transactions_export = transactions_df.copy()
            if 'date' in transactions_export.columns:
                if pd.api.types.is_datetime64_any_dtype(transactions_export['date']):
                    transactions_export['date'] = transactions_export['date'].dt.tz_localize(None)
            transactions_export.to_excel(writer, sheet_name='Transactions', index=False)
            
            # Add summary sheet
            summary_data = pd.DataFrame({
                'Metric': ['Total Value', 'Cash', 'Holdings Value', 'Total Return', 'Return %', 'Win Rate', 'Wins', 'Losses'],
                'Value': [
                    f"${portfolio_value['total_value']:,.2f}",
                    f"${portfolio_value['cash']:,.2f}",
                    f"${portfolio_value['holdings_value']:,.2f}",
                    f"${portfolio_value['total_return']:,.2f}",
                    f"{portfolio_value['total_return_pct']:.2f}%",
                    f"{metrics['win_rate']:.1f}%",
                    metrics['wins'],
                    metrics['losses']
                ]
            })
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        st.download_button(
            label="📥 Download Portfolio Report (Excel)",
            data=output.getvalue(),
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No transaction history yet. Start trading to see your history!")

# Tab 9: News Sentiment
if active_tab == 8:
    st.markdown('<h2 class="sub-header">📰 News Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze news sentiment for your selected stock using AI-powered sentiment analysis.
    """)
    
    # API Key input (optional)
    with st.expander("⚙️ NewsAPI Configuration (Optional)"):
        news_api_key = st.text_input(
            "NewsAPI Key",
            type="password",
            help="Enter your NewsAPI key from newsapi.org for real-time news. Leave blank for demo data."
        )
        if news_api_key:
            st.session_state['news_api_key'] = news_api_key
    
    if st.button("🔍 Analyze News Sentiment", type="primary"):
        with st.spinner("Fetching and analyzing news..."):
            try:
                # Initialize analyzer
                api_key = st.session_state.get('news_api_key', None)
                analyzer = NewsSentimentAnalyzer(api_key=api_key)
                
                # Get news sentiment
                sentiment_data = analyzer.get_news_sentiment(ticker)
                
                if sentiment_data and sentiment_data.get('articles') and len(sentiment_data.get('articles', [])) > 0:
                    # Display summary metrics
                    st.markdown("### 📊 Sentiment Overview")
                    
                    summary = sentiment_data['summary']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Articles",
                            summary['total_articles']
                        )
                    
                    with col2:
                        positive_pct = summary['positive_pct']
                        st.metric(
                            "Positive",
                            f"{summary['positive_count']}",
                            f"{positive_pct:.1f}%"
                        )
                    
                    with col3:
                        neutral_pct = summary['neutral_pct']
                        st.metric(
                            "Neutral",
                            f"{summary['neutral_count']}",
                            f"{neutral_pct:.1f}%"
                        )
                    
                    with col4:
                        negative_pct = summary['negative_pct']
                        st.metric(
                            "Negative",
                            f"{summary['negative_count']}",
                            f"{negative_pct:.1f}%"
                        )
                    
                    # Sentiment gauge
                    avg_sentiment = summary['avg_sentiment']
                    
                    if avg_sentiment > 0.1:
                        sentiment_label = "🟢 Positive"
                        sentiment_color = "#28a745"
                    elif avg_sentiment < -0.1:
                        sentiment_label = "🔴 Negative"
                        sentiment_color = "#dc3545"
                    else:
                        sentiment_label = "⚪ Neutral"
                        sentiment_color = "#6c757d"
                    
                    st.markdown(f"""
                    <div style='
                        text-align: center;
                        padding: 1.5rem;
                        background: linear-gradient(135deg, {sentiment_color}20, {sentiment_color}10);
                        border-radius: 15px;
                        border-left: 5px solid {sentiment_color};
                        margin: 1rem 0;
                    '>
                        <h3 style='margin: 0; color: {sentiment_color};'>
                            Overall Sentiment: {sentiment_label}
                        </h3>
                        <p style='font-size: 2rem; margin: 0.5rem 0; color: {sentiment_color};'>
                            {avg_sentiment:+.3f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display articles
                    st.markdown("### 📰 Recent News Articles")
                    
                    for i, article in enumerate(sentiment_data['articles'][:10], 1):
                        sentiment_info = article['sentiment']
                        sentiment = sentiment_info['polarity']
                        published_at = article.get('published', 'N/A')
                        
                        if sentiment > 0.1:
                            badge = "🟢 Positive"
                            badge_color = "#28a745"
                        elif sentiment < -0.1:
                            badge = "🔴 Negative"
                            badge_color = "#dc3545"
                        else:
                            badge = "⚪ Neutral"
                            badge_color = "#6c757d"
                        
                        st.markdown(f"""
                        <div style='
                            padding: 1.5rem;
                            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                            border-radius: 12px;
                            border-left: 5px solid {badge_color};
                            margin-bottom: 1.5rem;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                        '>
                            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;'>
                                <h4 style='margin: 0; color: #1a202c; font-weight: 600; font-size: 1.1rem; flex: 1;'>
                                    {i}. {article['title']}
                                </h4>
                                <span style='
                                    padding: 0.4rem 0.9rem;
                                    background: {badge_color};
                                    color: white;
                                    border-radius: 25px;
                                    font-size: 0.85rem;
                                    font-weight: 600;
                                    white-space: nowrap;
                                    margin-left: 1rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                '>
                                    {badge}
                                </span>
                            </div>
                            <p style='margin: 0.75rem 0; color: #2d3748; font-size: 1rem; line-height: 1.6;'>
                                {article['description']}
                            </p>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #e2e8f0;'>
                                <small style='color: #4a5568; font-weight: 500;'>{article['source']} • {published_at}</small>
                                <a href='{article['url']}' target='_blank' style='
                                    color: #667eea;
                                    text-decoration: none;
                                    font-size: 0.9rem;
                                '>Read More →</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Export option
                    st.markdown("### 💾 Export News Data")
                    
                    news_df = pd.DataFrame(sentiment_data['articles'])
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        news_df.to_excel(writer, sheet_name='News Articles', index=False)
                        
                        summary_df = pd.DataFrame({
                            'Metric': ['Total Articles', 'Positive', 'Neutral', 'Negative', 'Avg Sentiment'],
                            'Value': [
                                summary['total_articles'],
                                f"{summary['positive_count']} ({positive_pct:.1f}%)",
                                f"{summary['neutral_count']} ({neutral_pct:.1f}%)",
                                f"{summary['negative_count']} ({negative_pct:.1f}%)",
                                f"{avg_sentiment:+.3f}"
                            ]
                        })
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        label="📥 Download News Sentiment Report (Excel)",
                        data=output.getvalue(),
                        file_name=f"news_sentiment_{ticker}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                else:
                    st.warning("No news articles found for this ticker.")
                    
            except Exception as e:
                st.error(f"Error analyzing news sentiment: {str(e)}")
    else:
        st.info("👆 Click the button above to fetch and analyze news sentiment for the selected stock.")

# ==================== TAB 8: REAL-TIME MONITOR ====================
if active_tab == 8:
    st.markdown('<h2 class="sub-header">📡 Real-Time Stock Monitor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Monitor live stock prices with real-time updates, make instant predictions, and track intraday movements.
    """)
    
    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    try:
        # Fetch real-time quote and stock info
        with st.spinner("📡 Fetching real-time data..."):
            quote = fetcher.get_realtime_quote(ticker)
            stock_info = fetcher.get_stock_info(ticker)
        
        # ============ GOOGLE FINANCE STYLE LAYOUT ============
        
        # Top Section: Price Display
        price_col1, price_col2, price_col3 = st.columns([2, 2, 1])
        
        with price_col1:
            # Large price display
            change_symbol = "▲" if quote['change'] >= 0 else "▼"
            change_color = "#10b981" if quote['change'] >= 0 else "#ef4444"
            
            st.markdown(f"""
            <div style='padding: 1rem 0;'>
                <h1 style='margin: 0; font-size: 3rem; color: #e2e8f0;'>${quote['current_price']:.2f}
                <span style='font-size: 1.2rem; color: #94a3b8;'>USD</span>
                </h1>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; color: {change_color};'>
                    {change_symbol} ${abs(quote['change']):.2f} ({quote['change_percent']:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with price_col2:
            # Market status and timestamp
            from datetime import datetime
            now = datetime.now()
            market_closed_time = quote['timestamp'].strftime('%b %d, %I:%M %p')
            
            st.markdown(f"""
            <div style='padding: 1rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.9rem;'>
                    Closed: {market_closed_time} GMT-5 • <a href='#' style='color: #818cf8; text-decoration: none;'>Disclaimer</a>
                </p>
                <p style='margin: 0.5rem 0 0 0; color: #10b981; font-size: 0.95rem;'>
                    Pre-market {quote['current_price']:.2f} {quote['change_percent']:+.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with price_col3:
            # Refresh button
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
            auto_refresh = st.checkbox("Auto-Refresh (10s)", value=False)
        
        st.markdown("---")
        
        # Chart Period Selector (Google Finance style)
        period_options = {
            '1D': ('1d', '5m'),
            '5D': ('5d', '15m'),
            '1M': ('1mo', '1d'),
            '6M': ('6mo', '1d'),
            'YTD': ('ytd', '1d'),
            '1Y': ('1y', '1d'),
            '5Y': ('5y', '1wk'),
            'Max': ('max', '1wk')
        }
        
        selected_period = st.radio(
            "Period",
            options=list(period_options.keys()),
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        period, interval = period_options[selected_period]
        
        # Price Chart (Google Finance style)
        with st.spinner("📊 Loading chart..."):
            try:
                # Fetch data based on selected period
                if selected_period == '1D':
                    chart_df = fetcher.get_intraday_data(ticker, interval=interval, period=period)
                    time_col = 'Datetime'
                else:
                    chart_df = fetcher.fetch_stock_data(ticker, period=period, interval=interval)
                    # The fetcher already resets index, but column might be named differently
                    # Check for common date column names
                    if 'Date' in chart_df.columns:
                        time_col = 'Date'
                    elif 'Datetime' in chart_df.columns:
                        time_col = 'Datetime'
                    elif chart_df.index.name in ['Date', 'Datetime']:
                        chart_df = chart_df.reset_index()
                        time_col = chart_df.columns[0]  # First column after reset
                    else:
                        # Fallback: use the first column that looks like a date
                        date_cols = [col for col in chart_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if date_cols:
                            time_col = date_cols[0]
                        else:
                            # Last resort: reset index and use first column
                            chart_df = chart_df.reset_index()
                            time_col = chart_df.columns[0]
                
                # Verify we have data
                if len(chart_df) == 0:
                    st.warning("No data available for selected period")
                    st.stop()
                
                # Create Google Finance style chart
                fig = go.Figure()
                
                # Price line with gradient fill
                line_color = '#10b981' if quote['change'] >= 0 else '#ef4444'
                fill_color = 'rgba(16, 185, 129, 0.1)' if quote['change'] >= 0 else 'rgba(239, 68, 68, 0.1)'
                
                fig.add_trace(go.Scatter(
                    x=chart_df[time_col],
                    y=chart_df['Close'],
                    mode='lines',
                    name=ticker,
                    line=dict(color=line_color, width=2),
                    fill='tozeroy',
                    fillcolor=fill_color,
                    hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
                ))
                
                # Add current price line
                fig.add_hline(
                    y=quote['current_price'],
                    line_dash="dash",
                    line_color='#94a3b8',
                    line_width=1,
                    annotation_text=f"Current: ${quote['current_price']:.2f}",
                    annotation_position="right"
                )
                
                # Update layout to match Google Finance with auto-scaling y-axis
                y_min = chart_df['Close'].min() * 0.995
                y_max = chart_df['Close'].max() * 1.005
                
                fig.update_layout(
                    height=450,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(148, 163, 184, 0.1)',
                        showline=False,
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(148, 163, 184, 0.1)',
                        showline=False,
                        zeroline=False,
                        side='right',
                        range=[y_min, y_max]
                    ),
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    font=dict(color='#e2e8f0')
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Show data points info
                data_info_col1, data_info_col2 = st.columns(2)
                with data_info_col1:
                    st.caption(f"📊 Data points: {len(chart_df)}")
                with data_info_col2:
                    if len(chart_df) > 0:
                        latest_time = chart_df[time_col].iloc[-1]
                        if isinstance(latest_time, pd.Timestamp):
                            st.caption(f"Latest: {latest_time.strftime('%Y-%m-%d %H:%M')}")
                
            except Exception as e:
                st.error(f"❌ Error loading chart: {str(e)}")
                logger.error(f"Chart error: {str(e)}", exc_info=True)
                logger.error(f"Chart error: {str(e)}", exc_info=True)
        
        st.markdown("---")
        
        # Stock Statistics (Google Finance style)
        st.markdown("### 📊 Key Statistics")
        
        # Get additional info
        try:
            # Calculate 52-week high/low from yearly data
            yearly_data = fetcher.fetch_stock_data(ticker, period='1y', interval='1d')
            week_52_high = yearly_data['High'].max()
            week_52_low = yearly_data['Low'].min()
        except:
            week_52_high = 'N/A'
            week_52_low = 'N/A'
        
        # Display stats in Google Finance layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>Open</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>${quote['open']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>High</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>${quote['high']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>Low</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>${quote['low']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            market_cap_str = format_number(quote['market_cap']) if quote['market_cap'] != 'N/A' else 'N/A'
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>Mkt cap</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>{market_cap_str}</p>
            </div>
            """, unsafe_allow_html=True)
            
            pe_ratio = stock_info.get('pe_ratio', 'N/A')
            pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else 'N/A'
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>P/E ratio</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>{pe_str}</p>
            </div>
            """, unsafe_allow_html=True)
            
            week_52_high_str = f"${week_52_high:.2f}" if isinstance(week_52_high, (int, float)) else 'N/A'
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>52-wk high</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>{week_52_high_str}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>Prev close</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>${quote['previous_close']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>Volume</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>{quote['volume']:,}</p>
            </div>
            """, unsafe_allow_html=True)
            
            week_52_low_str = f"${week_52_low:.2f}" if isinstance(week_52_low, (int, float)) else 'N/A'
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>52-wk low</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>{week_52_low_str}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_vol_str = f"{quote['avg_volume']:,}" if quote['avg_volume'] != 'N/A' else 'N/A'
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>Avg volume</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>{avg_vol_str}</p>
            </div>
            """, unsafe_allow_html=True)
            
            day_range = quote['high'] - quote['low']
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <p style='margin: 0; color: #94a3b8; font-size: 0.85rem;'>Day range</p>
                <p style='margin: 0.2rem 0 0 0; color: #e2e8f0; font-size: 1rem; font-weight: 500;'>${quote['low']:.2f} - ${quote['high']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Empty div for spacing
            st.markdown("<div style='padding: 0.5rem 0;'></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Real-time prediction section (Google Finance style)
        if st.session_state.model_trained:
            st.markdown("""
            <div style='margin: 2rem 0 1rem 0;'>
                <h3 style='margin: 0; color: #e2e8f0; font-size: 1.25rem; font-weight: 600;'>Next Day Prediction</h3>
                <p style='margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.875rem;'>AI forecast based on historical patterns and technical indicators</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prediction parameters
            with st.expander("⚙️ Prediction Settings", expanded=False):
                param_col1, param_col2, param_col3 = st.columns(3)
                
                with param_col1:
                    pred_seq_length = st.number_input(
                        "Sequence Length",
                        min_value=10,
                        max_value=120,
                        value=seq_length,
                        step=10,
                        help="Number of historical days to use for prediction"
                    )
                
                with param_col2:
                    pred_data_period = st.selectbox(
                        "Historical Data Period",
                        options=['1y', '2y', '5y', 'max'],
                        index=1,
                        help="Amount of historical data to fetch"
                    )
                
                with param_col3:
                    selected_model = st.selectbox(
                        "Model to Use",
                        options=['LSTM', 'Attention-LSTM', 'N-BEATS', 'TCN', 'Transformer', 'Ensemble'],
                        index=0,
                        help="Select which model to use for prediction"
                    )
            
            if st.button("⚡ Make Prediction on Current Data", type="primary"):
                with st.spinner("Generating real-time prediction..."):
                    try:
                        # Fetch recent data for prediction with fallback
                        try:
                            recent_data = fetcher.fetch_stock_data(ticker, period=pred_data_period, interval='1d')
                        except (TypeError, AttributeError) as e:
                            # If 'max' period fails, fall back to 5y
                            if pred_data_period == 'max':
                                st.warning("⚠️ Unable to fetch maximum historical data. Falling back to 5 years of data.")
                                recent_data = fetcher.fetch_stock_data(ticker, period='5y', interval='1d')
                            else:
                                raise
                        
                        # Add features
                        feature_engineer = FeatureEngineer()
                        recent_data = feature_engineer.add_all_indicators(recent_data)
                        
                        # Drop NaN values and check if we have enough data
                        recent_data = recent_data.dropna()
                        
                        if len(recent_data) == 0:
                            st.error("❌ Not enough data after processing. The stock may be newly listed or have insufficient history.")
                            st.info("💡 **Tip:** This stock may require more historical data. Try stocks with longer trading history.")
                            st.stop()
                        
                        if len(recent_data) < pred_seq_length:
                            st.warning(f"""
                            ⚠️ **Insufficient Data for Prediction**
                            
                            - Found: {len(recent_data)} valid data points
                            - Required: {pred_seq_length} data points
                            
                            **Reason:** After calculating technical indicators (MA, RSI, MACD) and removing invalid values, 
                            there isn't enough historical data for the model to make a reliable prediction.
                            
                            **Solutions:**
                            - Try a different stock with longer trading history
                            - Reduce sequence length in Prediction Settings above
                            - Increase historical data period to '5y' or 'max'
                            - Wait for more trading days to accumulate data
                            """)
                            st.stop()
                        
                        # Prepare data
                        features = recent_data[config.FEATURES].values
                        
                        # Verify features are not empty
                        if len(features) == 0:
                            st.error("❌ No valid feature data available after processing.")
                            st.stop()
                        
                        normalized_features = st.session_state.preprocessor.feature_scaler.transform(features)
                        
                        # Create sequence
                        if len(normalized_features) >= pred_seq_length:
                            last_sequence = normalized_features[-pred_seq_length:]
                            last_sequence = last_sequence.reshape(1, pred_seq_length, len(config.FEATURES))
                            
                            # Convert to tensor for PyTorch models
                            import torch
                            
                            # Get prediction from selected model
                            model_map = {
                                'LSTM': 'lstm_model',
                                'Attention-LSTM': 'attention_lstm_model',
                                'N-BEATS': 'nbeats_model',
                                'TCN': 'tcn_model',
                                'Transformer': 'transformer_model',
                                'Ensemble': 'ensemble_model'
                            }
                            
                            model_key = model_map.get(selected_model)
                            
                            if model_key and model_key in st.session_state:
                                model = st.session_state[model_key]
                            else:
                                st.error(f"❌ {selected_model} model not found. Please train the model first.")
                                st.stop()
                            
                            # Move tensor to same device as model (CPU or CUDA)
                            device = next(model.model.parameters()).device
                            last_sequence_tensor = torch.FloatTensor(last_sequence).to(device)
                            
                            model.model.eval()
                            with torch.no_grad():
                                prediction = model.model(last_sequence_tensor).cpu().numpy()
                            
                            # Inverse transform
                            predicted_price = st.session_state.preprocessor.inverse_transform_target(prediction)[0][0]
                            
                            # Display prediction (Google Finance style)
                            current_price = quote['current_price']
                            price_change = predicted_price - current_price
                            price_change_pct = (price_change / current_price) * 100
                            
                            # Create clean prediction card
                            change_color = "#34d399" if price_change >= 0 else "#ef4444"
                            change_symbol = "▲" if price_change >= 0 else "▼"
                            
                            st.markdown(f"""
                            <div style='background: #1e293b; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;'>
                                    <div>
                                        <p style='margin: 0; color: #94a3b8; font-size: 0.875rem;'>Predicted Close (Next Day)</p>
                                        <p style='margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 2.5rem; font-weight: 700;'>${predicted_price:.2f}</p>
                                        <p style='margin: 0.5rem 0 0 0; color: {change_color}; font-size: 1.125rem; font-weight: 600;'>
                                            {change_symbol} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)
                                        </p>
                                    </div>
                                    <div style='text-align: right;'>
                                        <p style='margin: 0; color: #94a3b8; font-size: 0.875rem;'>Current Price</p>
                                        <p style='margin: 0.5rem 0 0 0; color: #e2e8f0; font-size: 1.5rem; font-weight: 600;'>${current_price:.2f}</p>
                                    </div>
                                </div>
                                <div style='border-top: 1px solid #334155; padding-top: 1rem;'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <div>
                                            <p style='margin: 0; color: #94a3b8; font-size: 0.875rem;'>Signal</p>
                                            <p style='margin: 0.25rem 0 0 0; color: #e2e8f0; font-size: 1.125rem; font-weight: 600;'>
                                                {"BUY" if price_change_pct > 1 else "SELL" if price_change_pct < -1 else "HOLD"}
                                            </p>
                                        </div>
                                        <div style='text-align: right;'>
                                            <p style='margin: 0; color: #94a3b8; font-size: 0.875rem;'>Model</p>
                                            <p style='margin: 0.25rem 0 0 0; color: #e2e8f0; font-size: 1.125rem; font-weight: 600;'>
                                                {selected_model}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.warning(f"Not enough data points. Need at least {pred_seq_length} days of data.")
                    
                    except Exception as e:
                        st.error(f"Error making real-time prediction: {str(e)}")
                        logger.error(f"Real-time prediction error: {str(e)}", exc_info=True)
        else:
            st.markdown("""
            <div style='background: #1e293b; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;'>
                <p style='margin: 0; color: #94a3b8; font-size: 1rem;'>Train a model first to enable next-day predictions</p>
                <p style='margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.875rem;'>Navigate to the Model Training tab to get started</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Auto-refresh logic
        if auto_refresh:
            import time
            time.sleep(10)
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error fetching real-time data: {str(e)}")
        logger.error(f"Real-time monitor error: {str(e)}", exc_info=True)
        st.info("💡 **Tip:** Real-time data is only available during market hours. For stocks outside US markets, availability may vary.")

# ==================== TAB 9: RISK METRICS ====================
if active_tab == 9:
    st.markdown('<h2 class="sub-header">⚠️ Risk Metrics Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Comprehensive risk analysis including VaR, CVaR, Sharpe Ratio, Maximum Drawdown, and more.
    """)
    
    if st.button("📊 Calculate Risk Metrics", type="primary"):
        with st.spinner("Calculating risk metrics..."):
            try:
                fetcher = StockDataFetcher()
                
                # Fetch historical data (1 year)
                risk_data = fetcher.fetch_stock_data(ticker, period='1y')
                
                if risk_data is not None and not risk_data.empty and len(risk_data) > 30:
                    # Get comprehensive metrics
                    metrics = RiskAnalyzer.get_comprehensive_risk_metrics(risk_data)
                    risk_rating_info = RiskAnalyzer.get_risk_rating(metrics)
                    risk_rating = risk_rating_info['rating']
                    rating_color = risk_rating_info['color']
                    
                    st.markdown(f"""
                    <div style='
                        text-align: center;
                        padding: 2rem;
                        background: linear-gradient(135deg, {rating_color}20, {rating_color}10);
                        border-radius: 15px;
                        border: 3px solid {rating_color};
                        margin: 1rem 0;
                    '>
                        <h2 style='margin: 0; color: {rating_color};'>
                            Risk Rating: {risk_rating}
                        </h2>
                        <p style='font-size: 1.2rem; margin: 0.5rem 0; color: {rating_color};'>
                            Score: {risk_rating_info['score']}/{risk_rating_info['total']} ({risk_rating_info['percentage']:.1f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics in columns
                    st.markdown("### 📊 Value at Risk (VaR)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "VaR (95% confidence)",
                            f"{metrics['var_95_pct']:+.2f}%",
                            help="Maximum expected loss with 95% confidence over 1 day"
                        )
                    
                    with col2:
                        st.metric(
                            "VaR (99% confidence)",
                            f"{metrics['var_99_pct']:+.2f}%",
                            help="Maximum expected loss with 99% confidence over 1 day"
                        )
                    
                    st.markdown("### 💀 Conditional Value at Risk (CVaR)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "CVaR (95%)",
                            f"{metrics['cvar_95_pct']:+.2f}%",
                            help="Expected loss given that VaR threshold is exceeded"
                        )
                    
                    with col2:
                        st.metric(
                            "CVaR (99%)",
                            f"{metrics['cvar_99_pct']:+.2f}%",
                            help="Expected loss in the worst 1% of scenarios"
                        )
                    
                    st.markdown("### 📉 Maximum Drawdown")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Max Drawdown",
                            f"{metrics['max_drawdown_pct']:.2f}%",
                            help="Largest peak-to-trough decline"
                        )
                    
                    with col2:
                        # Calculate duration if recovery date exists
                        if metrics.get('recovery_date') and metrics.get('peak_date'):
                            try:
                                duration = (pd.to_datetime(metrics['recovery_date']) - pd.to_datetime(metrics['peak_date'])).days
                            except:
                                duration = 'N/A'
                        else:
                            duration = 'Ongoing'
                        
                        st.metric(
                            "Drawdown Duration",
                            f"{duration} days" if isinstance(duration, int) else duration,
                            help="Days from peak to recovery"
                        )
                    
                    with col3:
                        peak_date = metrics.get('peak_date', 'N/A')
                        trough_date = metrics.get('trough_date', 'N/A')
                        st.info(f"**Peak:** {peak_date}  \n**Trough:** {trough_date}")
                    
                    # Plot drawdown
                    fig_dd = go.Figure()
                    
                    # Use drawdown series from metrics
                    drawdown = metrics['drawdown_series'] * 100
                    
                    fig_dd.add_trace(go.Scatter(
                        x=risk_data.index,
                        y=drawdown,
                        fill='tozeroy',
                        fillcolor='rgba(220, 53, 69, 0.3)',
                        line=dict(color='#dc3545', width=2),
                        name='Drawdown'
                    ))
                    
                    fig_dd.update_layout(
                        title="Drawdown Over Time",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        template="plotly_white",
                        hovermode='x unified',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_dd, width='stretch')
                    
                    st.markdown("### 📈 Performance Ratios")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        sharpe = metrics.get('sharpe_ratio')
                        st.metric(
                            "Sharpe Ratio",
                            f"{sharpe:.3f}" if sharpe is not None else "N/A",
                            help="Risk-adjusted return (higher is better)"
                        )
                    
                    with col2:
                        sortino = metrics.get('sortino_ratio')
                        st.metric(
                            "Sortino Ratio",
                            f"{sortino:.3f}" if sortino is not None else "N/A",
                            help="Return relative to downside risk"
                        )
                    
                    with col3:
                        calmar = metrics.get('calmar_ratio')
                        st.metric(
                            "Calmar Ratio",
                            f"{calmar:.3f}" if calmar is not None else "N/A",
                            help="Return relative to maximum drawdown"
                        )
                    
                    with col4:
                        beta_value = metrics.get('beta')
                        st.metric(
                            "Beta",
                            f"{beta_value:.3f}" if beta_value is not None else "N/A",
                            help="Volatility relative to market (S&P 500)"
                        )
                    
                    st.markdown("### 📊 Volatility Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Daily Volatility",
                            f"{metrics['volatility']:.4f}",
                            help="Standard deviation of daily returns"
                        )
                    
                    with col2:
                        st.metric(
                            "Annualized Volatility",
                            f"{metrics['annualized_volatility_pct']:.2f}%",
                            help="Volatility scaled to annual terms"
                        )
                    
                    # Export risk metrics
                    st.markdown("### 💾 Export Risk Report")
                    
                    # Calculate duration safely
                    if metrics.get('recovery_date') and metrics.get('peak_date'):
                        try:
                            duration_val = (pd.to_datetime(metrics['recovery_date']) - pd.to_datetime(metrics['peak_date'])).days
                            duration_str = f"{duration_val} days"
                        except:
                            duration_str = 'Ongoing'
                    else:
                        duration_str = 'Ongoing'
                    
                    risk_metrics_df = pd.DataFrame({
                        'Metric': [
                            'VaR (95%)', 'VaR (99%)',
                            'CVaR (95%)', 'CVaR (99%)',
                            'Max Drawdown',
                            'Drawdown Duration',
                            'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                            'Beta', 'Daily Volatility', 'Annualized Volatility',
                            'Risk Rating'
                        ],
                        'Value': [
                            f"{metrics['var_95_pct']:+.2f}%", 
                            f"{metrics['var_99_pct']:+.2f}%",
                            f"{metrics['cvar_95_pct']:+.2f}%", 
                            f"{metrics['cvar_99_pct']:+.2f}%",
                            f"{metrics['max_drawdown_pct']:.2f}%",
                            duration_str,
                            f"{metrics['sharpe_ratio']:.3f}", 
                            f"{metrics['sortino_ratio']:.3f}",
                            f"{metrics['calmar_ratio']:.3f}", 
                            f"{metrics['beta']:.3f}" if metrics.get('beta') is not None else 'N/A',
                            f"{metrics['volatility']:.4f}",
                            f"{metrics['annualized_volatility_pct']:.2f}%",
                            risk_rating
                        ]
                    })
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        risk_metrics_df.to_excel(writer, sheet_name='Risk Metrics', index=False)
                        
                        # Add drawdown data
                        dd_data = pd.DataFrame({
                            'Date': risk_data.index,
                            'Drawdown (%)': drawdown.values
                        })
                        if pd.api.types.is_datetime64_any_dtype(dd_data['Date']):
                            dd_data['Date'] = dd_data['Date'].dt.tz_localize(None)
                        dd_data.to_excel(writer, sheet_name='Drawdown History', index=False)
                    
                    st.download_button(
                        label="📥 Download Risk Report (Excel)",
                        data=output.getvalue(),
                        file_name=f"risk_metrics_{ticker}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                else:
                    st.error("Insufficient data to calculate risk metrics. Need at least 30 days of historical data.")
                    
            except Exception as e:
                st.error(f"Error calculating risk metrics: {str(e)}")
    else:
        st.info("👆 Click the button above to calculate comprehensive risk metrics for the selected stock.")

# Tab 11: Pattern Recognition
if active_tab == 10:
    st.markdown('<h2 class="sub-header">🔍 Pattern Recognition</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Detect candlestick patterns and technical chart patterns to identify potential trading opportunities.
    """)
    
    if st.button("🔎 Detect Patterns", type="primary"):
        with st.spinner("Analyzing patterns..."):
            try:
                fetcher = StockDataFetcher()
                
                # Fetch historical data (6 months)
                pattern_data = fetcher.fetch_stock_data(ticker, period='6mo')
                
                if pattern_data is not None and not pattern_data.empty and len(pattern_data) > 20:
                    # Detect all patterns
                    patterns_result = PatternRecognizer.detect_all_patterns(pattern_data)
                    
                    # Safely extract patterns
                    patterns = patterns_result.get('patterns', {})
                    if not isinstance(patterns, dict):
                        st.error("Invalid pattern data format")
                        st.stop()
                    
                    # Get overall signal
                    signal_result = PatternRecognizer.get_pattern_signal(patterns_result)
                    
                    # Display signal
                    signal_colors = {
                        'BULLISH': '#28a745',
                        'BEARISH': '#dc3545',
                        'NEUTRAL': '#6c757d'
                    }
                    
                    signal_color = signal_colors.get(signal_result['signal'], '#6c757d')
                    
                    st.markdown(f"""
                    <div style='
                        text-align: center;
                        padding: 2rem;
                        background: linear-gradient(135deg, {signal_color}20, {signal_color}10);
                        border-radius: 15px;
                        border: 3px solid {signal_color};
                        margin: 1rem 0;
                    '>
                        <h2 style='margin: 0 0 0.5rem 0; color: {signal_color};'>
                            Pattern Signal: {signal_result['signal']}
                        </h2>
                        <p style='font-size: 1.5rem; margin: 0; color: {signal_color};'>
                            Strength: {signal_result['strength']:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display pattern counts
                    st.markdown("### 📊 Candlestick Patterns Detected (Last 30 Days)")
                    
                    # Helper function to safely get pattern count
                    def get_pattern_count(pattern_name):
                        pattern_data = patterns.get(pattern_name, {})
                        if isinstance(pattern_data, dict):
                            return pattern_data.get('count', 0)
                        return 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("#### 🟢 Bullish Patterns")
                        st.metric("Hammer", get_pattern_count('Hammer'))
                        st.metric("Bullish Engulfing", get_pattern_count('Bullish Engulfing'))
                        st.metric("Morning Star", get_pattern_count('Morning Star'))
                    
                    with col2:
                        st.markdown("#### 🔴 Bearish Patterns")
                        st.metric("Shooting Star", get_pattern_count('Shooting Star'))
                        st.metric("Bearish Engulfing", get_pattern_count('Bearish Engulfing'))
                        st.metric("Evening Star", get_pattern_count('Evening Star'))
                    
                    with col3:
                        st.markdown("#### ⚪ Neutral Patterns")
                        st.metric("Doji", get_pattern_count('Doji'))
                    
                    with col4:
                        st.markdown("#### 📈 Summary")
                        total_bullish = (get_pattern_count('Hammer') + 
                                       get_pattern_count('Bullish Engulfing') + 
                                       get_pattern_count('Morning Star'))
                        total_bearish = (get_pattern_count('Shooting Star') + 
                                       get_pattern_count('Bearish Engulfing') + 
                                       get_pattern_count('Evening Star'))
                        st.metric("Total Bullish", total_bullish)
                        st.metric("Total Bearish", total_bearish)
                    
                    st.markdown("---")
                    
                    # Support and Resistance Levels
                    st.markdown("### 📍 Support & Resistance Levels")
                    
                    sr_levels = patterns_result.get('support_resistance', {})
                    
                    if sr_levels and isinstance(sr_levels, dict) and 'support' in sr_levels and 'resistance' in sr_levels:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🟢 Support Levels")
                            support_levels = sr_levels.get('support', [])
                            if isinstance(support_levels, (list, np.ndarray)):
                                for i, level in enumerate(list(support_levels)[:5], 1):
                                    if isinstance(level, (int, float, np.number)):
                                        st.info(f"**S{i}:** ${float(level):.2f}")
                            else:
                                st.warning("Invalid support levels format")
                        
                        with col2:
                            st.markdown("#### 🔴 Resistance Levels")
                            resistance_levels = sr_levels.get('resistance', [])
                            if isinstance(resistance_levels, (list, np.ndarray)):
                                for i, level in enumerate(list(resistance_levels)[:5], 1):
                                    if isinstance(level, (int, float, np.number)):
                                        st.info(f"**R{i}:** ${float(level):.2f}")
                            else:
                                st.warning("Invalid resistance levels format")
                        
                        # Plot price with S/R levels
                        fig_sr = go.Figure()
                        
                        # Add candlestick chart
                        fig_sr.add_trace(go.Candlestick(
                            x=pattern_data.index[-60:],
                            open=pattern_data['Open'][-60:],
                            high=pattern_data['High'][-60:],
                            low=pattern_data['Low'][-60:],
                            close=pattern_data['Close'][-60:],
                            name='Price'
                        ))
                        
                        # Add support levels
                        support_levels = sr_levels.get('support', [])
                        if isinstance(support_levels, (list, np.ndarray)):
                            for level in list(support_levels)[:3]:
                                if isinstance(level, (int, float, np.number)):
                                    fig_sr.add_hline(
                                        y=float(level),
                                        line_dash="dash",
                                        line_color="green",
                                        annotation_text=f"S: ${float(level):.2f}",
                                        annotation_position="right"
                                    )
                        
                        # Add resistance levels
                        resistance_levels = sr_levels.get('resistance', [])
                        if isinstance(resistance_levels, (list, np.ndarray)):
                            for level in list(resistance_levels)[:3]:
                                if isinstance(level, (int, float, np.number)):
                                    fig_sr.add_hline(
                                        y=float(level),
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"R: ${float(level):.2f}",
                                        annotation_position="right"
                                    )
                        
                        fig_sr.update_layout(
                            title=f"{ticker} - Support & Resistance Levels (Last 60 Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_white",
                            xaxis_rangeslider_visible=False,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_sr, width='stretch')
                    else:
                        st.warning("No significant support/resistance levels detected.")
                    
                    # Pattern explanation
                    st.markdown("### 📚 Pattern Explanations")
                    
                    with st.expander("🟢 Bullish Patterns"):
                        st.markdown("""
                        - **Hammer**: Small body at top, long lower shadow. Indicates potential reversal from downtrend.
                        - **Bullish Engulfing**: Large green candle fully engulfs previous red candle. Strong bullish reversal.
                        - **Morning Star**: Three-candle pattern (bearish, small, bullish). Signals bottom reversal.
                        """)
                    
                    with st.expander("🔴 Bearish Patterns"):
                        st.markdown("""
                        - **Shooting Star**: Small body at bottom, long upper shadow. Indicates potential reversal from uptrend.
                        - **Bearish Engulfing**: Large red candle fully engulfs previous green candle. Strong bearish reversal.
                        - **Evening Star**: Three-candle pattern (bullish, small, bearish). Signals top reversal.
                        """)
                    
                    with st.expander("⚪ Neutral Patterns"):
                        st.markdown("""
                        - **Doji**: Open and close are nearly equal. Indicates indecision and potential reversal.
                        """)
                    
                    # Export pattern data
                    st.markdown("### 💾 Export Pattern Analysis")
                    
                    # Create pattern dataframe with safe access
                    pattern_list = []
                    pattern_names = [
                        ('Doji', 'Neutral'),
                        ('Hammer', 'Bullish'),
                        ('Shooting Star', 'Bearish'),
                        ('Bullish Engulfing', 'Bullish'),
                        ('Bearish Engulfing', 'Bearish'),
                        ('Morning Star', 'Bullish'),
                        ('Evening Star', 'Bearish')
                    ]
                    
                    for pattern_name, pattern_type in pattern_names:
                        count = patterns.get(pattern_name, {}).get('count', 0) if isinstance(patterns.get(pattern_name), dict) else 0
                        pattern_list.append({
                            'Pattern': pattern_name,
                            'Count (30 days)': count,
                            'Type': pattern_type
                        })
                    
                    pattern_df = pd.DataFrame(pattern_list)
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        pattern_df.to_excel(writer, sheet_name='Pattern Summary', index=False)
                        
                        # Add S/R levels - safely handle the data
                        if sr_levels and isinstance(sr_levels, dict):
                            try:
                                support_vals = sr_levels.get('support', [])
                                resistance_vals = sr_levels.get('resistance', [])
                                
                                # Convert to list and ensure numeric values
                                if isinstance(support_vals, (list, np.ndarray)):
                                    support_list = [float(x) for x in list(support_vals)[:10] if isinstance(x, (int, float, np.number))]
                                else:
                                    support_list = []
                                    
                                if isinstance(resistance_vals, (list, np.ndarray)):
                                    resistance_list = [float(x) for x in list(resistance_vals)[:10] if isinstance(x, (int, float, np.number))]
                                else:
                                    resistance_list = []
                                
                                if support_list or resistance_list:
                                    sr_df = pd.DataFrame({
                                        'Support Levels': pd.Series(support_list),
                                        'Resistance Levels': pd.Series(resistance_list)
                                    })
                                    sr_df.to_excel(writer, sheet_name='Support Resistance', index=False)
                            except Exception as e:
                                # Skip S/R export if there's an error
                                pass
                        
                        # Add signal summary
                        signal_df = pd.DataFrame({
                            'Metric': ['Signal', 'Strength', 'Total Bullish', 'Total Bearish'],
                            'Value': [
                                signal_result['signal'],
                                f"{signal_result['strength']:.1f}%",
                                total_bullish,
                                total_bearish
                            ]
                        })
                        signal_df.to_excel(writer, sheet_name='Signal Summary', index=False)
                    
                    st.download_button(
                        label="📥 Download Pattern Analysis (Excel)",
                        data=output.getvalue(),
                        file_name=f"pattern_analysis_{ticker}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                else:
                    st.error("Insufficient data for pattern recognition. Need at least 20 days of historical data.")
                    
            except Exception as e:
                st.error(f"Error detecting patterns: {str(e)}")
    else:
        st.info("👆 Click the button above to detect candlestick patterns and support/resistance levels.")

# Compact Footer
st.markdown("---")
st.markdown("""
<div style='
    text-align: center; 
    padding: 1.5rem; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    color: white;
    margin-top: 2rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
'>
    <h3 style='margin: 0 0 0.5rem 0; font-family: Poppins, sans-serif;'>📈 Stock Price Predictor with LSTM</h3>
    <p style='font-size: 0.95rem; margin: 0;'>
        Built with ❤️ using <strong>Streamlit</strong>, <strong>TensorFlow</strong> & <strong>Keras</strong>
    </p>
</div>
""", unsafe_allow_html=True)
