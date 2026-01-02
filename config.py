"""Configuration file for Stock Price Prediction with LSTM"""

# Model Hyperparameters
SEQ_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32
LSTM_UNITS = [50, 50]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Data Parameters
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# Features - Enhanced with already-calculated indicators
FEATURES = ['Close', 'Volume', 'MA_50', 'MA_200', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Returns', 'Volume_Ratio']
TARGET = 'Close'

# Technical Indicators
MA_WINDOWS = [50, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Data Source
DEFAULT_TICKER = 'AAPL'
DATA_PERIOD = '5y'
DATA_INTERVAL = '1d'

# Model Paths
MODEL_DIR = 'models/'
MODEL_PATH = 'models/lstm_stock_model.pth'
SCALER_PATH = 'models/scaler.pkl'

# Plotting
FIGURE_SIZE = (14, 7)
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Streamlit Config (ADD THESE LINES)
PAGE_TITLE = "📈 Stock Price Predictor with LSTM"
PAGE_ICON = "📈"
LAYOUT = "wide"