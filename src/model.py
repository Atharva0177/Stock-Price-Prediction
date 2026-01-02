"""
LSTM Model Module
Build and configure LSTM neural network using PyTorch with GPU support
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable cuDNN benchmarking for better GPU performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    logger.info("GPU optimization enabled: cuDNN benchmarking activated")


class LSTMNet(nn.Module):
    """PyTorch LSTM Network"""
    
    def __init__(self, n_features, lstm_units=[50, 50], dropout_rate=0.2):
        super(LSTMNet, self).__init__()
        
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=n_features, 
                            hidden_size=lstm_units[0], 
                            batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=lstm_units[0], 
                            hidden_size=lstm_units[1], 
                            batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_units[1], 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
    
    def forward(self, x):
        # First LSTM
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Take only the last time step
        x = x[:, -1, :]
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


class LSTMModel: 
    """Build and manage PyTorch LSTM model for stock price prediction"""
    
    def __init__(self, seq_length, n_features, lstm_units=[50, 50], 
                 dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize LSTM model
        
        Args: 
            seq_length (int): Length of input sequences
            n_features (int): Number of features
            lstm_units (list): Units in each LSTM layer
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Log detailed GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 Using GPU: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.2f} GB")
            logger.info(f"⚡ CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("⚠️  GPU not available, using CPU (training will be slower)")
    
    def build_model(self):
        """
        Build LSTM model architecture
        
        Returns:
            LSTMNet: PyTorch LSTM model
        """
        logger.info("Building PyTorch LSTM model...")
        
        self.model = LSTMNet(
            n_features=self.n_features,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Successfully built LSTM model on {self.device}")
        logger.info(f"Model has {total_params:,} trainable parameters")
        
        return self.model
    
    def get_optimizer(self):
        """Get optimizer for training"""
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def get_criterion(self):
        """Get loss criterion"""
        return nn.MSELoss()
    
    def get_callbacks(self, model_path='models/best_model.h5', patience=10):
        """
        Get training callbacks (for compatibility with existing code)
        Note: PyTorch doesn't use Keras callbacks, but we keep this for interface compatibility
        
        Args: 
            model_path (str): Path to save best model
            patience (int): Early stopping patience
            
        Returns: 
            dict: Callback configuration
        """
        return {
            'model_path': model_path,
            'patience': patience
        }
    
    def save_model(self, filepath='models/lstm_model.pth'):
        """
        Save model to disk
        
        Args: 
            filepath (str): Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model_from_file(self, filepath='models/lstm_model.pth'):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore configuration
        self.seq_length = checkpoint['seq_length']
        self.n_features = checkpoint['n_features']
        self.lstm_units = checkpoint['lstm_units']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        
        # Rebuild and load weights
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns: 
            str: Model summary
        """
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            summary = f"""
PyTorch LSTM Model Summary:
==========================
LSTM Units: {self.lstm_units}
Dropout Rate: {self.dropout_rate}
Sequence Length: {self.seq_length}
Number of Features: {self.n_features}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Device: {self.device}
"""
            return summary
        else:
            return "Model not built yet"