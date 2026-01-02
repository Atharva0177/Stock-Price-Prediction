"""
Ensemble Model
Combines predictions from multiple models for better accuracy
"""

import numpy as np
import torch
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of multiple models with weighted averaging"""
    
    def __init__(self, models, weights=None, method='weighted_average'):
        """
        Initialize ensemble model
        
        Args:
            models (list): List of trained model instances
            weights (list): Weights for each model (must sum to 1)
            method (str): Ensemble method ('weighted_average', 'stacking', 'voting')
        """
        self.models = models
        self.method = method
        
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights
        
        logger.info(f"Ensemble model initialized with {len(models)} models")
        logger.info(f"Weights: {self.weights}")
        logger.info(f"Method: {method}")
    
    def predict(self, X):
        """
        Make ensemble predictions
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Ensemble predictions
        """
        predictions = []
        
        # Get predictions from each model
        for i, (model, trainer) in enumerate(self.models):
            try:
                pred = trainer.predict(X)
                predictions.append(pred)
                logger.info(f"Got predictions from model {i+1}/{len(self.models)}")
            except Exception as e:
                logger.error(f"Error getting predictions from model {i+1}: {e}")
                # Use zeros as placeholder if model fails
                predictions.append(np.zeros((X.shape[0], 1)))
        
        # Convert to array
        predictions = np.array(predictions)  # (n_models, n_samples, 1)
        
        # Ensemble method
        if self.method == 'weighted_average':
            # Weighted average of predictions
            ensemble_pred = np.zeros_like(predictions[0])
            for i, weight in enumerate(self.weights):
                ensemble_pred += weight * predictions[i]
            
        elif self.method == 'median':
            # Median of predictions (robust to outliers)
            ensemble_pred = np.median(predictions, axis=0)
            
        elif self.method == 'min_variance':
            # Weight inversely proportional to prediction variance
            variances = np.var(predictions, axis=1, keepdims=True)
            inv_var = 1.0 / (variances + 1e-6)
            weights_dynamic = inv_var / np.sum(inv_var)
            ensemble_pred = np.sum(weights_dynamic * predictions, axis=0)
            
        else:
            # Default to simple average
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble model
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating ensemble model...")
        
        # Get ensemble predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
        
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
        
        logger.info("Ensemble evaluation completed")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        return metrics
    
    def optimize_weights(self, X_val, y_val):
        """
        Optimize ensemble weights using validation data
        
        Args:
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
        """
        from scipy.optimize import minimize
        
        logger.info("Optimizing ensemble weights...")
        
        # Get predictions from all models
        predictions = []
        for i, (model, trainer) in enumerate(self.models):
            pred = trainer.predict(X_val)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Objective function: minimize MSE
        def objective(weights):
            ensemble_pred = np.sum(weights.reshape(-1, 1, 1) * predictions, axis=0)
            mse = mean_squared_error(y_val, ensemble_pred)
            return mse
        
        # Constraints: weights sum to 1, all non-negative
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # Initial weights
        x0 = np.array(self.weights)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            self.weights = result.x.tolist()
            logger.info(f"Optimized weights: {self.weights}")
            logger.info(f"Validation MSE: {result.fun:.6f}")
        else:
            logger.warning("Weight optimization failed, keeping original weights")
    
    def get_model_summary(self):
        """Get ensemble model summary"""
        summary = f"""
Ensemble Model Summary:
======================
Number of Models: {len(self.models)}
Ensemble Method: {self.method}
Model Weights: {[f'{w:.3f}' for w in self.weights]}

Individual Models:
"""
        for i, (model, _) in enumerate(self.models):
            model_name = model.__class__.__name__
            summary += f"\n{i+1}. {model_name} (weight: {self.weights[i]:.3f})"
        
        return summary
