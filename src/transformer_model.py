"""
Transformer Model for Time Series
Simplified Temporal Fusion Transformer architecture
"""

import torch
import torch.nn as nn
import math
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerNet(nn.Module):
    """Transformer Network for Time Series"""
    
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=3, dropout=0.2):
        super(TransformerNet, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x: (batch, seq, features)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Take last time step
        x = x[:, -1, :]
        
        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerModel:
    """Transformer model wrapper"""
    
    def __init__(self, seq_length, n_features, d_model=64, nhead=4,
                 num_layers=3, dropout_rate=0.2, learning_rate=0.0005):
        """
        Initialize Transformer model
        
        Args:
            seq_length (int): Length of input sequences
            n_features (int): Number of features
            d_model (int): Dimension of transformer
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
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
        """Build Transformer model"""
        logger.info("Building PyTorch Transformer model...")
        
        self.model = TransformerNet(
            n_features=self.n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout_rate
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Successfully built Transformer model on {self.device}")
        logger.info(f"Model has {total_params:,} trainable parameters")
        
        return self.model
    
    def get_optimizer(self):
        """Get optimizer with warmup scheduling"""
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                                weight_decay=1e-5, betas=(0.9, 0.98), eps=1e-9)
    
    def get_criterion(self):
        """Get loss criterion"""
        return nn.MSELoss()
    
    def get_callbacks(self, model_path='models/best_model.pth', patience=10):
        """Get callbacks configuration"""
        return {'model_path': model_path, 'patience': patience}
    
    def save_model(self, filepath='models/transformer_model.pth'):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model_from_file(self, filepath='models/transformer_model.pth'):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.seq_length = checkpoint['seq_length']
        self.n_features = checkpoint['n_features']
        self.d_model = checkpoint['d_model']
        self.nhead = checkpoint['nhead']
        self.num_layers = checkpoint['num_layers']
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
PyTorch Transformer Model Summary:
=================================
Model Dimension: {self.d_model}
Attention Heads: {self.nhead}
Transformer Layers: {self.num_layers}
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
