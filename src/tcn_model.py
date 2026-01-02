"""
Temporal Convolutional Network (TCN) Model
Fast and efficient temporal modeling with dilated convolutions
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable cuDNN benchmarking
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolutions"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.padding = padding
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Apply causal padding (pad left side only)
        x_padded = nn.functional.pad(x, (self.padding, 0))
        
        # First convolution
        out = self.conv1(x_padded)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second convolution with padding
        out_padded = nn.functional.pad(out, (self.padding, 0))
        out = self.conv2(out_padded)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Ensure out and res have the same length
        if out.size(2) != res.size(2):
            # Crop to minimum length
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Removes the extra padding from the right side"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TCNNet(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(self, n_features, num_channels=[32, 32, 64, 64], kernel_size=3, dropout=0.2):
        super(TCNNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                    dilation=dilation_size,
                                    padding=(kernel_size-1) * dilation_size,
                                    dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        
        # Apply TCN
        y = self.network(x)
        
        # Take last time step
        y = y[:, :, -1]
        
        # Final prediction
        return self.fc(y)


class TCNModel:
    """TCN model wrapper"""
    
    def __init__(self, seq_length, n_features, num_channels=[32, 32, 64, 64],
                 kernel_size=3, dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize TCN model
        
        Args:
            seq_length (int): Length of input sequences
            n_features (int): Number of features
            num_channels (list): Channels in each TCN level
            kernel_size (int): Kernel size for convolutions
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 Using GPU: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.2f} GB")
        else:
            logger.warning("⚠️  GPU not available, using CPU")
    
    def build_model(self):
        """Build TCN model"""
        logger.info("Building PyTorch TCN model...")
        
        self.model = TCNNet(
            n_features=self.n_features,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout_rate
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Successfully built TCN model on {self.device}")
        logger.info(f"Model has {total_params:,} trainable parameters")
        
        return self.model
    
    def get_optimizer(self):
        """Get optimizer"""
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
    
    def get_criterion(self):
        """Get loss criterion"""
        return nn.MSELoss()
    
    def get_callbacks(self, model_path='models/best_model.pth', patience=10):
        """Get callbacks configuration"""
        return {'model_path': model_path, 'patience': patience}
    
    def save_model(self, filepath='models/tcn_model.pth'):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model_from_file(self, filepath='models/tcn_model.pth'):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.seq_length = checkpoint['seq_length']
        self.n_features = checkpoint['n_features']
        self.num_channels = checkpoint['num_channels']
        self.kernel_size = checkpoint['kernel_size']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            summary = f"""
PyTorch TCN Model Summary:
=========================
Channels: {self.num_channels}
Kernel Size: {self.kernel_size}
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
