"""
Model Training Module
Train and evaluate PyTorch LSTM model with GPU support
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


class ModelTrainer:
    """Train and evaluate PyTorch LSTM model"""
    
    def __init__(self, model, use_amp=True):
        """
        Initialize trainer
        
        Args:
            model: LSTM model instance
            use_amp: Use automatic mixed precision for GPU training
        """
        self.model = model
        self.history = None
        self.device = model.device
        self.optimizer = model.get_optimizer()
        self.criterion = model.get_criterion()
        
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
             validation_split=0.1, callbacks=None):
        """
        Train the model
        
        Args: 
            X_train (np.array): Training features
            y_train (np.array): Training targets
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
            callbacks (dict): Training callback configuration (contains patience, model_path)
            
        Returns:
            dict: Training history
        """
        logger.info(f"Starting PyTorch LSTM training for {epochs} epochs...")
        
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
        
        # Training history
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        
        # Early stopping setup
        patience = callbacks.get('patience', 10) if callbacks else 10
        model_path = callbacks.get('model_path', 'models/best_model.pth') if callbacks else 'models/best_model.pth'
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.model.train()
            train_losses = []
            train_maes = []
            
            for X_batch, y_batch in train_loader:
                # Move batches to GPU
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                if self.use_amp:
                    with autocast('cuda'):
                        y_pred = self.model.model(X_batch)
                        loss = self.criterion(y_pred, y_batch)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision
                    y_pred = self.model.model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                
                train_losses.append(loss.item())
                # Calculate MAE
                mae = torch.mean(torch.abs(y_pred - y_batch)).item()
                train_maes.append(mae)
            
            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)
            history['loss'].append(avg_train_loss)
            history['mae'].append(avg_train_mae)
            
            # Validation phase
            self.model.model.eval()
            val_losses = []
            val_maes = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    # Move batches to GPU
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    y_pred = self.model.model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    val_losses.append(loss.item())
                    
                    # Calculate MAE
                    mae = torch.mean(torch.abs(y_pred - y_batch)).item()
                    val_maes.append(mae)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_mae = np.mean(val_maes)
            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(avg_val_mae)
            
            # Print progress with GPU memory usage
            if (epoch + 1) % 10 == 0 or epoch == 0:
                gpu_mem_info = ""
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    gpu_mem_info = f" | GPU Mem: {gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f} GB"
                
                logger.info(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_train_loss:.6f}, MAE: {avg_train_mae:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}{gpu_mem_info}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.model.save_model(model_path)
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    # Load best model
                    self.model.load_model_from_file(model_path)
                    break
        
        # Create history object with both dictionary access and attribute access
        class History:
            def __init__(self, hist_dict):
                self.history = hist_dict
        
        self.history = History(history)
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
            predictions = self.model.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
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
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Calculate direction accuracy
        y_test_direction = np.diff(y_test.flatten()) > 0
        y_pred_direction = np.diff(y_pred.flatten()) > 0
        direction_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
        
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
        Predict future prices
        
        Args:
            last_sequence (np.array): Last sequence of data
            n_days (int): Number of days to predict
            preprocessor: DataPreprocessor instance
            
        Returns:
            np.array: Future predictions
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        self.model.model.eval()
        
        for _ in range(n_days):
            # Predict next value
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                next_pred = self.model.model(seq_tensor)
                next_pred = next_pred.cpu().numpy()[0, 0]
            
            predictions.append(next_pred)
            
            # Update sequence - use the predicted value for the close price
            # and repeat the last values for other features
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred  # Assuming Close is the first feature
            
            # Shift sequence and add new prediction
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform to get actual prices
        predictions_actual = preprocessor.inverse_transform_target(predictions)
        
        return predictions_actual
        
        return predictions_actual