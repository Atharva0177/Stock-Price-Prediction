"""
N-BEATS Model Module
Build and configure N-BEATS neural network for stock price prediction
Based on the paper: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
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


class NBeatsBlock(nn.Module):
    """
    Basic building block of N-BEATS
    Each block contains fully connected layers for backcast and forecast
    """
    
    def __init__(self, units, thetas_dim, device, backcast_length, forecast_length, share_thetas=False):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.device = device
        
        # Fully connected layers
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        
        # Theta layers for backcast and forecast
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)
    
    def forward(self, x):
        # Fully connected stack
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        
        # Theta parameters
        return x


class GenericBlock(NBeatsBlock):
    """Generic block for N-BEATS (default mode without interpretability)"""
    
    def __init__(self, units, thetas_dim, device, backcast_length, forecast_length):
        super().__init__(units, thetas_dim, device, backcast_length, forecast_length)
        
        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)
    
    def forward(self, x):
        x = super().forward(x)
        
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)
        
        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)
        
        return backcast, forecast


class TrendBlock(NBeatsBlock):
    """Trend block for interpretable N-BEATS"""
    
    def __init__(self, units, thetas_dim, device, backcast_length, forecast_length, nb_harmonics=None):
        super().__init__(units, thetas_dim, device, backcast_length, forecast_length, share_thetas=True)
        
        self.backcast_time = torch.linspace(0, 1, backcast_length).to(device)
        self.forecast_time = torch.linspace(0, 1, forecast_length).to(device)
    
    def forward(self, x):
        x = super().forward(x)
        theta = self.theta_f_fc(x)
        
        backcast = self._trend_model(theta, self.backcast_time, self.backcast_length)
        forecast = self._trend_model(theta, self.forecast_time, self.forecast_length)
        
        return backcast, forecast
    
    def _trend_model(self, theta, t, length):
        # Polynomial trend model
        p = theta.size(1)  # Number of polynomial coefficients
        batch_size = theta.size(0)
        
        # Create time basis: shape (batch, length)
        T = t.unsqueeze(0).repeat(batch_size, 1)
        
        # Create polynomial basis: shape (batch, length, p)
        powers = torch.arange(p, dtype=torch.float32).to(self.device)
        T_expanded = T.unsqueeze(-1)  # (batch, length, 1)
        powers_expanded = powers.unsqueeze(0).unsqueeze(0)  # (1, 1, p)
        polynomial_basis = T_expanded ** powers_expanded  # (batch, length, p)
        
        # Compute trend: (batch, length, p) * (batch, p) -> (batch, length)
        trend = torch.einsum('blp,bp->bl', polynomial_basis, theta)
        return trend


class SeasonalityBlock(NBeatsBlock):
    """Seasonality block for interpretable N-BEATS"""
    
    def __init__(self, units, thetas_dim, device, backcast_length, forecast_length, nb_harmonics=None):
        super().__init__(units, thetas_dim, device, backcast_length, forecast_length, share_thetas=True)
        
        if nb_harmonics is None:
            nb_harmonics = forecast_length // 2
        self.nb_harmonics = nb_harmonics
        
        self.backcast_time = 2 * np.pi * torch.arange(backcast_length, dtype=torch.float32).to(device) / backcast_length
        self.forecast_time = 2 * np.pi * torch.arange(forecast_length, dtype=torch.float32).to(device) / forecast_length
    
    def forward(self, x):
        x = super().forward(x)
        theta = self.theta_f_fc(x)
        
        backcast = self._seasonality_model(theta, self.backcast_time)
        forecast = self._seasonality_model(theta, self.forecast_time)
        
        return backcast, forecast
    
    def _seasonality_model(self, theta, t):
        # Fourier series for seasonality
        p = theta.size(1)
        batch_size = theta.size(0)
        assert p == 2 * self.nb_harmonics, f"Expected theta size {2 * self.nb_harmonics}, got {p}"
        
        s = torch.zeros(batch_size, len(t)).to(self.device)
        for i in range(self.nb_harmonics):
            # theta[:, 2*i] is (batch,), cos/sin(i*t) is (length,)
            # Need to reshape for broadcasting: (batch, 1) * (length,) -> (batch, length)
            cos_term = theta[:, 2*i].unsqueeze(1) * torch.cos(i * t)
            sin_term = theta[:, 2*i + 1].unsqueeze(1) * torch.sin(i * t)
            s += cos_term + sin_term
        
        return s


class NBeatsNet(nn.Module):
    """
    N-BEATS neural network for time series forecasting
    Can be configured as Generic or Interpretable (Trend + Seasonality)
    """
    
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'
    
    def __init__(self, device, stack_types=(GENERIC_BLOCK, GENERIC_BLOCK),
                 nb_blocks_per_stack=3, forecast_length=7, backcast_length=30,
                 thetas_dim=(4, 8), share_weights_in_stack=False,
                 hidden_layer_units=128, nb_harmonics=None):
        """
        Initialize N-BEATS model
        
        Args:
            device: torch device (cpu or cuda)
            stack_types: Types of stacks (generic, trend, seasonality)
            nb_blocks_per_stack: Number of blocks per stack
            forecast_length: Number of time steps to forecast
            backcast_length: Number of time steps to use as input
            thetas_dim: Dimensions for theta parameters
            share_weights_in_stack: Whether to share weights within stack
            hidden_layer_units: Number of hidden units in each layer
            nb_harmonics: Number of harmonics for seasonality block
        """
        super().__init__()
        
        self.device = device
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.thetas_dim = thetas_dim
        self.parameters_list = []
        
        # Build stacks
        self.stacks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            self.stacks.append(self._create_stack(stack_id, stack_type))
        
        self.to(device)
    
    def _create_stack(self, stack_id, stack_type):
        """Create a stack of blocks"""
        stack = nn.ModuleList()
        
        for block_id in range(self.nb_blocks_per_stack):
            if self.share_weights_in_stack and block_id > 0:
                # Share weights within stack
                block = stack[-1]
            else:
                # Create new block
                if stack_type == self.GENERIC_BLOCK:
                    block = GenericBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=self.thetas_dim[stack_id],
                        device=self.device,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length
                    )
                elif stack_type == self.TREND_BLOCK:
                    block = TrendBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=self.thetas_dim[stack_id],
                        device=self.device,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length
                    )
                elif stack_type == self.SEASONALITY_BLOCK:
                    # Calculate nb_harmonics from thetas_dim
                    nb_harmonics = self.thetas_dim[stack_id] // 2
                    block = SeasonalityBlock(
                        units=self.hidden_layer_units,
                        thetas_dim=self.thetas_dim[stack_id],
                        device=self.device,
                        backcast_length=self.backcast_length,
                        forecast_length=self.forecast_length,
                        nb_harmonics=nb_harmonics
                    )
                else:
                    raise ValueError(f"Unknown block type: {stack_type}")
            
            stack.append(block)
        
        return stack
    
    def forward(self, backcast):
        """
        Forward pass through N-BEATS
        
        Args:
            backcast: Input tensor of shape (batch_size, backcast_length)
            
        Returns:
            backcast: Reconstructed backcast
            forecast: Predicted forecast
        """
        # Ensure backcast has correct shape
        if len(backcast.shape) == 3:
            # If input is (batch, seq, features), take last time step of each feature
            backcast = backcast[:, -1, :]
        elif len(backcast.shape) == 2:
            # Already (batch, backcast_length)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {backcast.shape}")
        
        # Ensure we have exactly backcast_length features
        if backcast.shape[1] != self.backcast_length:
            # If we have more features, take the first backcast_length
            if backcast.shape[1] > self.backcast_length:
                backcast = backcast[:, :self.backcast_length]
            else:
                # Pad with zeros if we have fewer
                padding = torch.zeros(backcast.shape[0], self.backcast_length - backcast.shape[1]).to(self.device)
                backcast = torch.cat([backcast, padding], dim=1)
        
        forecast = torch.zeros(backcast.size(0), self.forecast_length).to(self.device)
        
        for stack in self.stacks:
            for block in stack:
                b, f = block(backcast)
                backcast = backcast - b
                forecast = forecast + f
        
        return backcast, forecast


class NBeatsModel:
    """Wrapper class for N-BEATS model to match LSTM interface"""
    
    def __init__(self, seq_length, n_features, forecast_length=1,
                 hidden_layer_units=128, stack_types=('generic', 'generic'),
                 nb_blocks_per_stack=3, learning_rate=0.001):
        """
        Initialize N-BEATS model wrapper
        
        Args:
            seq_length (int): Length of input sequences (backcast_length)
            n_features (int): Number of features (not directly used, for interface compatibility)
            forecast_length (int): Number of steps to forecast
            hidden_layer_units (int): Units in hidden layers
            stack_types (tuple): Types of stacks to use
            nb_blocks_per_stack (int): Number of blocks per stack
            learning_rate (float): Learning rate for optimizer
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.forecast_length = forecast_length
        self.hidden_layer_units = hidden_layer_units
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
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
        Build N-BEATS model
        
        Returns:
            NBeatsNet: Compiled N-BEATS model
        """
        logger.info("Building N-BEATS model...")
        
        # Determine thetas_dim based on stack types
        thetas_dim = []
        for stack_type in self.stack_types:
            if stack_type == 'generic':
                # For generic, use smaller theta dimension for regularization
                thetas_dim.append(min(self.seq_length, 128))
            elif stack_type == 'trend':
                # Polynomial degree - higher order for complex trends
                thetas_dim.append(8)  # Increased from 4
            elif stack_type == 'seasonality':
                # Number of harmonics * 2 - more harmonics for complex patterns
                thetas_dim.append(16)  # Increased from 8
        
        self.model = NBeatsNet(
            device=self.device,
            stack_types=self.stack_types,
            nb_blocks_per_stack=self.nb_blocks_per_stack,
            forecast_length=self.forecast_length,
            backcast_length=self.seq_length,
            thetas_dim=tuple(thetas_dim),
            hidden_layer_units=self.hidden_layer_units,
            share_weights_in_stack=False
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"✓ Successfully built N-BEATS model")
        logger.info(f"Model has {total_params:,} trainable parameters")
        
        return self.model
    
    def get_optimizer(self):
        """Get optimizer for training with weight decay for regularization"""
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
    
    def get_criterion(self):
        """Get loss criterion"""
        return nn.MSELoss()
    
    def save_model(self, filepath='models/nbeats_model.pth'):
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
            'forecast_length': self.forecast_length,
            'hidden_layer_units': self.hidden_layer_units,
            'stack_types': self.stack_types,
            'nb_blocks_per_stack': self.nb_blocks_per_stack,
            'learning_rate': self.learning_rate
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model_from_file(self, filepath='models/nbeats_model.pth'):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore configuration
        self.seq_length = checkpoint['seq_length']
        self.n_features = checkpoint['n_features']
        self.forecast_length = checkpoint['forecast_length']
        self.hidden_layer_units = checkpoint['hidden_layer_units']
        self.stack_types = checkpoint['stack_types']
        self.nb_blocks_per_stack = checkpoint['nb_blocks_per_stack']
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
N-BEATS Model Summary:
=====================
Stack Types: {self.stack_types}
Blocks per Stack: {self.nb_blocks_per_stack}
Backcast Length: {self.seq_length}
Forecast Length: {self.forecast_length}
Hidden Units: {self.hidden_layer_units}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Device: {self.device}
"""
            return summary
        else:
            return "Model not built yet"
