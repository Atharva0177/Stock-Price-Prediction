# N-BEATS Integration Guide

## Overview

This project now includes **N-BEATS (Neural Basis Expansion Analysis for Time Series)** as an alternative to the LSTM model for stock price prediction. N-BEATS is a deep learning model specifically designed for time series forecasting that offers interpretability and robust performance without relying on recurrent connections.

## What is N-BEATS?

N-BEATS was developed by researchers at Element AI and MILA. It revolutionizes time series forecasting by using:

- **Feed-forward Architecture**: Unlike RNNs, N-BEATS uses stacked fully-connected layers
- **Interpretability**: Can decompose predictions into trend and seasonality components
- **Modularity**: Configurable blocks that can be stacked for different use cases
- **No Domain-Specific Feature Engineering**: Learns patterns directly from data

## Key Features

### Architecture Components

1. **Generic Blocks**: Default mode for general-purpose forecasting
2. **Trend Blocks**: Captures polynomial trends in the data (interpretable mode)
3. **Seasonality Blocks**: Captures periodic patterns using Fourier series (interpretable mode)

### How It Works

- **Backcasting**: Reconstructs past values to ensure the model understands historical patterns
- **Forecasting**: Predicts future values by aggregating outputs from all blocks
- **Residual Learning**: Each block refines predictions by addressing residuals from previous blocks

## Model Selection in the App

You can now choose between two models in the Streamlit app:

1. **LSTM** (Original)
   - Recurrent neural network
   - Good for sequential dependencies
   - Uses CuDNN acceleration when available

2. **N-BEATS** (New)
   - Feed-forward architecture
   - Interpretable forecasting
   - Better generalization on diverse time series

## Installation

The required dependencies have been added to `requirements.txt`:

```bash
pip install torch>=2.0.0
pip install nbeats-pytorch>=1.8.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### In the Streamlit App

1. Launch the app: `streamlit run app.py`
2. In the sidebar, under "Model Parameters", select:
   - **Model Type**: Choose between "LSTM" or "N-BEATS"
3. Configure other parameters (sequence length, epochs, batch size)
4. Click "🚀 Train Model" to train the selected model

### Model Files

- LSTM model: `models/lstm_model.h5` or `models/best_model.h5`
- N-BEATS model: `models/nbeats_model.pth`
- Scalers: `models/feature_scaler.pkl`, `models/target_scaler.pkl`

## Technical Implementation

### Files Added

1. **src/nbeats_model.py**: Core N-BEATS implementation
   - `NBeatsBlock`: Base block class
   - `GenericBlock`: General-purpose block
   - `TrendBlock`: Interpretable trend component
   - `SeasonalityBlock`: Interpretable seasonality component
   - `NBeatsNet`: Full N-BEATS network
   - `NBeatsModel`: Wrapper class for interface compatibility

2. **src/nbeats_trainer.py**: Training and evaluation
   - `NBeatsTrainer`: Handles training loop, evaluation, and future predictions
   - Compatible with existing preprocessing pipeline
   - Early stopping and validation support

### Key Differences from LSTM

| Feature | LSTM | N-BEATS |
|---------|------|---------|
| Architecture | Recurrent (RNN) | Feed-forward |
| Interpretability | Limited | High (with interpretable blocks) |
| Training Stability | Gradient issues possible | More stable |
| GPU Acceleration | CuDNN required for speed | Standard PyTorch ops |
| Input Processing | Sequences with features | Univariate lookback window |
| Output | Single value prediction | Backcast + forecast |

## Hyperparameters

### Configurable in the App
- **Sequence Length (Backcast Length)**: Number of past days (30-90)
- **Epochs**: Training iterations (10-100)
- **Batch Size**: Samples per batch (16, 32, 64, 128)

### Fixed in Code (can be modified in app.py)
- **Hidden Layer Units**: 128 (neurons in each layer)
- **Stack Types**: ('generic', 'generic') - two generic stacks
- **Blocks per Stack**: 3
- **Learning Rate**: From config.LEARNING_RATE

## Hyperparameter Tuning

Based on the article, you can tune:

1. **Backcast/Forecast Length**: Longer for capturing more patterns
2. **Hidden Layer Units**: 64, 128, 256 (more = higher capacity)
3. **Number of Blocks**: 2, 3, 4 per stack
4. **Stack Types**: 
   - Generic: `('generic', 'generic')`
   - Interpretable: `('trend', 'seasonality')`
   - Mixed: `('generic', 'trend')`
5. **Learning Rate**: 0.001, 0.01, 0.1
6. **Number of Harmonics**: For seasonality blocks (default: forecast_length // 2)

## Benefits for Stock Prediction

1. **Robust to Noise**: N-BEATS handles market volatility well
2. **No Recurrent Connections**: Avoids vanishing/exploding gradient problems
3. **Interpretable Components**: Can separate trend from seasonality
4. **Fast Training**: No sequential dependencies during training
5. **Better Generalization**: Works across different stocks without retuning

## Comparison with LSTM

Both models are now available in the project:

- **Use LSTM** when:
  - You need to capture long-term sequential dependencies
  - Feature engineering is extensive
  - You have access to GPU with CuDNN support

- **Use N-BEATS** when:
  - You want interpretable predictions
  - Training stability is important
  - You're working with diverse stocks/markets
  - You need faster training on CPU

## Future Enhancements

Potential improvements:
- Add interpretable mode (trend + seasonality blocks)
- Ensemble predictions (LSTM + N-BEATS)
- Multi-horizon forecasting
- Attention mechanisms
- Custom block architectures

## References

- Original Paper: [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)
- Article: [N-BEATS: The Unique Interpretable Deep Learning Model for Time Series Forecasting](https://medium.com/@captnitinbhatnagar/n-beats-the-unique-interpretable-deep-learning-model-for-time-series-forecasting-8dfdefaf0e34)
- GitHub Implementation: [n-beats on GitHub](https://github.com/philipperemy/n-beats)

## Support

Both models share the same preprocessing pipeline, evaluation metrics, and visualization tools, making it easy to compare performance and switch between them.

For questions or issues with N-BEATS integration, check the implementation in:
- `src/nbeats_model.py` - Model architecture
- `src/nbeats_trainer.py` - Training logic
- `app.py` - UI integration (search for `model_type`)
