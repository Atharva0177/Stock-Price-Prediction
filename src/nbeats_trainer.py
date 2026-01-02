"""
N-BEATS Model Training Module
Train and evaluate N-BEATS model for stock price prediction
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mixed precision training for GPU performance
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


class NBeatsTrainer:
    """Train and evaluate N-BEATS model"""
    
    def __init__(self, model, use_amp=True):
        """
        Initialize trainer
        
        Args:
            model: NBeatsModel instance
            use_amp: Use automatic mixed precision for GPU training
        """
        self.model = model
        self.history = {'train_loss': [], 'val_loss': []}
        self.device = model.device
        self.optimizer = model.get_optimizer()
        self.criterion = model.get_criterion()
        
        # Learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Setup mixed precision training for GPU
        self.use_amp = use_amp and AMP_AVAILABLE and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            logger.info("⚡ Mixed precision training enabled (faster GPU training)")
        else:
            self.scaler = None
            if torch.cuda.is_available() and not AMP_AVAILABLE:
                logger.warning("Mixed precision not available, using standard precision")
    
    def train(self, X_train, y_train, epochs=50, batch_size=32,
              validation_split=0.1, patience=10, callbacks=None):
        """
        Train the N-BEATS model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            patience (int): Early stopping patience
            callbacks: Not used (for interface compatibility)
            
        Returns:
            dict: Training history
        """
        logger.info(f"Starting N-BEATS training for {epochs} epochs...")
        
        # Convert to PyTorch tensors (keep on CPU for DataLoader)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Split into train and validation
        n_samples = len(X_train)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        X_train_split = X_train_tensor[:n_train]
        y_train_split = y_train_tensor[:n_train]
        X_val_split = X_train_tensor[n_train:]
        y_val_split = y_train_tensor[n_train:]
        
        # Create data loaders with pin_memory for faster GPU transfer
        pin_memory = torch.cuda.is_available()
        train_dataset = TensorDataset(X_train_split, y_train_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 pin_memory=pin_memory, num_workers=0)
        
        val_dataset = TensorDataset(X_val_split, y_val_split)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               pin_memory=pin_memory, num_workers=0)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                # Move batches to GPU
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                if self.use_amp:
                    with autocast('cuda'):
                        _, forecast = self.model.model(X_batch)
                        
                        # Ensure forecast matches target shape
                        if forecast.shape[1] != y_batch.shape[1]:
                            forecast = forecast[:, :y_batch.shape[1]]
                        
                        loss = self.criterion(forecast, y_batch)
                    
                    # Backward pass with gradient scaling and clipping
                    self.scaler.scale(loss).backward()
                    # Gradient clipping for stability
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision
                    _, forecast = self.model.model(X_batch)
                    
                    # Ensure forecast matches target shape
                    if forecast.shape[1] != y_batch.shape[1]:
                        forecast = forecast[:, :y_batch.shape[1]]
                    
                    loss = self.criterion(forecast, y_batch)
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    # Move batches to GPU
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    _, forecast = self.model.model(X_batch)
                    
                    # Ensure forecast matches target shape
                    if forecast.shape[1] != y_batch.shape[1]:
                        forecast = forecast[:, :y_batch.shape[1]]
                    
                    loss = self.criterion(forecast, y_batch)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            self.history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Print progress with GPU memory usage
            if (epoch + 1) % 10 == 0 or epoch == 0:
                gpu_mem_info = ""
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    gpu_mem_info = f" | GPU Mem: {gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f} GB"
                
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}{gpu_mem_info}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.model.state_dict()
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    # Restore best model
                    self.model.model.load_state_dict(self.best_model_state)
                    break
        
        logger.info("Training completed")
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Predictions
        """
        self.model.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, forecast = self.model.model(X_tensor)
            predictions = forecast.cpu().numpy()
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating N-BEATS model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Ensure shapes match
        if y_pred.shape[1] != y_test.shape[1]:
            y_pred = y_pred[:, :y_test.shape[1]]
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_test != 0
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = 0.0
        
        # Calculate direction accuracy
        if y_test.shape[0] > 1:
            y_test_direction = np.diff(y_test.flatten()) > 0
            y_pred_direction = np.diff(y_pred.flatten()) > 0
            direction_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
        else:
            direction_accuracy = 0.0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info("Evaluation completed")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        return metrics
    
    def predict_future(self, last_sequence, n_days, preprocessor):
        """
        Predict future prices using N-BEATS
        
        Args:
            last_sequence (np.array): Last sequence of data
            n_days (int): Number of days to predict
            preprocessor: DataPreprocessor instance
            
        Returns:
            np.array: Future predictions
        """
        self.model.model.eval()
        predictions = []
        
        # Get the backcast length from the model
        backcast_length = self.model.seq_length
        
        # Convert to tensor
        current_sequence = torch.FloatTensor(last_sequence).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_days):
                # Reshape for model input if needed
                if len(current_sequence.shape) == 2:
                    # (seq_length, n_features) -> (1, seq_length, n_features)
                    input_seq = current_sequence.unsqueeze(0)
                else:
                    input_seq = current_sequence
                
                # Predict next value
                _, forecast = self.model.model(input_seq)
                next_pred = forecast[0, 0].item()  # Get first forecast value
                predictions.append(next_pred)
                
                # Update sequence
                # Create new row with predicted value for close price
                new_row = current_sequence[-1].clone()
                new_row[0] = next_pred  # Assuming Close is the first feature
                
                # Shift sequence and add new prediction
                current_sequence = torch.cat([current_sequence[1:], new_row.unsqueeze(0)], dim=0)
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform to get actual prices
        predictions_actual = preprocessor.inverse_transform_target(predictions)
        
        return predictions_actual
    
    def get_training_history(self):
        """
        Get training history
        
        Returns:
            dict: Training history with loss values
        """
        return self.history
