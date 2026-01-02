"""
Data Preprocessing Module
Normalize data and create sequences for LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor: 
    """Preprocess data for LSTM model"""
    
    def __init__(self, feature_columns, target_column='Close'):
        """
        Initialize preprocessor
        
        Args: 
            feature_columns (list): List of feature column names
            target_column (str): Target column name
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
    
    def normalize_data(self, df):
        """
        Normalize features and target using MinMaxScaler
        
        Args:
            df (pd. DataFrame): Data to normalize
            
        Returns: 
            tuple: (normalized_features, normalized_target, original_df)
        """
        # Extract features and target
        features = df[self.feature_columns]. values
        target = df[[self.target_column]].values
        
        # Fit and transform
        normalized_features = self.feature_scaler.fit_transform(features)
        normalized_target = self. target_scaler.fit_transform(target)
        
        logger.info(f"Data normalized.  Features shape: {normalized_features. shape}")
        
        return normalized_features, normalized_target, df
    
    def inverse_transform_target(self, scaled_target):
        """
        Inverse transform target values to original scale
        
        Args: 
            scaled_target (np. array): Scaled target values
            
        Returns:
            np.array: Original scale values
        """
        return self.target_scaler.inverse_transform(scaled_target)
    
    def create_sequences(self, features, target, seq_length=60):
        """
        Create sequences for LSTM input
        
        Args:
            features (np.array): Feature data
            target (np.array): Target data
            seq_length (int): Sequence length (number of timesteps)
            
        Returns: 
            tuple: (X, y) - sequences and targets
        """
        X, y = [], []
        
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences.  X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets
        
        Args:
            X (np. array): Feature sequences
            y (np.array): Target values
            test_size (float): Proportion of data for testing
            
        Returns: 
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_scalers(self, feature_path='models/feature_scaler.pkl', 
                    target_path='models/target_scaler.pkl'):
        """
        Save fitted scalers to disk
        
        Args: 
            feature_path (str): Path to save feature scaler
            target_path (str): Path to save target scaler
        """
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(target_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)
        logger.info("Scalers saved successfully")
    
    def load_scalers(self, feature_path='models/feature_scaler.pkl', 
                    target_path='models/target_scaler.pkl'):
        """
        Load fitted scalers from disk
        
        Args:
            feature_path (str): Path to feature scaler
            target_path (str): Path to target scaler
        """
        with open(feature_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.target_scaler = pickle.load(f)
        logger.info("Scalers loaded successfully")