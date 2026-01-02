"""
Attention-LSTM Model Module
LSTM with attention mechanism for improved feature focus
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


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        return context, attention_weights


class AttentionLSTMNet(nn.Module):
    """PyTorch Attention-LSTM Network"""
    
    def __init__(self, n_features, lstm_units=[64, 64], dropout_rate=0.2):
        super(AttentionLSTMNet, self).__init__()
        
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=n_features, 
                            hidden_size=lstm_units[0], 
                            batch_first=True,
                            dropout=dropout_rate if len(lstm_units) > 1 else 0)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=lstm_units[0], 
                            hidden_size=lstm_units[1], 
                            batch_first=True)
        
        # Attention layer
        self.attention = AttentionLayer(lstm_units[1])
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(lstm_units[1], 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # First LSTM
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        
        # Second LSTM
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        # Attention mechanism
        x, attention_weights = self.attention(x)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class AttentionLSTMModel:
    """Attention-LSTM model wrapper"""
    
    def __init__(self, seq_length, n_features, lstm_units=[64, 64], 
                 dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize Attention-LSTM model
        
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
        """Build Attention-LSTM model"""
        logger.info("Building PyTorch Attention-LSTM model...")
        
        self.model = AttentionLSTMNet(
            n_features=self.n_features,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Successfully built Attention-LSTM model on {self.device}")
        logger.info(f"Model has {total_params:,} trainable parameters")
        
        return self.model
    
    def get_optimizer(self):
        """Get optimizer for training"""
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
    
    def get_criterion(self):
        """Get loss criterion"""
        return nn.MSELoss()
    
    def get_callbacks(self, model_path='models/best_model.pth', patience=10):
        """Get training callbacks configuration"""
        return {
            'model_path': model_path,
            'patience': patience
        }
    
    def save_model(self, filepath='models/attention_lstm_model.pth'):
        """Save model to disk"""
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
    
    def load_model_from_file(self, filepath='models/attention_lstm_model.pth'):
        """Load model from disk"""
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
        """Get model architecture summary"""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            summary = f"""
PyTorch Attention-LSTM Model Summary:
====================================
LSTM Units: {self.lstm_units}
Dropout Rate: {self.dropout_rate}
Sequence Length: {self.seq_length}
Number of Features: {self.n_features}
Attention Mechanism: Enabled
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Device: {self.device}
"""
            return summary
        else:
            return "Model not built yet"
