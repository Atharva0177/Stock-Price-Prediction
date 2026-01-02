# Quick Start Guide - Advanced Models

## How to Use the New Models

### 1. **Run the Streamlit App**
```bash
streamlit run app.py
```

### 2. **Select a Model**
In the sidebar under "🤖 Model Parameters":
- Choose from: LSTM, Attention-LSTM, N-BEATS, TCN, Transformer, or Ensemble

### 3. **Configure Settings**
- **Sequence Length**: 30-90 days (default: 60)
- **Epochs**: 20-200 (default: 50)
- **Batch Size**: 16-128 (default: 32)
- **🔥 Train All Models & Compare**: Enable to train all 5 models and see performance comparison

### 4. **Train the Model**
#### Option A: Train Single Model
- Click "🚀 Train Model"
- Watch real-time progress (5 steps)
- View training metrics

#### Option B: Train All Models (Comparison Mode) 🔥
- Check "🔥 Train All Models & Compare" in sidebar
- Click "🚀 Start Training All Models"
- Wait for all 5 models to train (~20-30 minutes total)
- View comprehensive comparison:
  - Performance metrics table
  - R² Score chart
  - RMSE comparison chart
  - Training time chart
  - Automatic recommendations

### 5. **Evaluate Results**
- Model architecture summary
- Training history plot
- Performance metrics (R², RMSE, MAE)
- Predictions vs actual prices

## Quick Comparison

### Speed Priority → Use **TCN** or **LSTM**
- Training: 2-3 minutes
- Good accuracy
- Low memory usage

### Accuracy Priority → Use **Ensemble** or **Transformer**
- Training: 10-12 minutes
- Best accuracy
- Requires more GPU memory

### Interpretability Priority → Use **N-BEATS**
- Shows trend and seasonality components
- Explainable predictions
- Training: 10-15 minutes

### Balanced Choice → Use **Attention-LSTM**
- Better than LSTM
- Faster than Ensemble
- Shows feature importance

## Model Files Generated

After training, you'll find these files in `models/`:

```
models/
├── lstm_stock_model.pth           # LSTM model
├── attention_lstm_model.pth       # Attention-LSTM model
├── nbeats_model.pth               # N-BEATS model
├── tcn_model.pth                  # TCN model
├── transformer_model.pth          # Transformer model
├── ensemble_lstmmodel.pth         # Ensemble LSTM component
├── ensemble_attentionlstmmodel.pth # Ensemble Attention-LSTM component
├── ensemble_tcnmodel.pth          # Ensemble TCN component
├── ensemble_weights.json          # Ensemble weights
├── feature_scaler.pkl             # Feature normalization
└── target_scaler.pkl              # Target normalization
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size (try 16 instead of 32)
- Reduce sequence length (try 45 instead of 60)
- Use smaller model (LSTM, TCN instead of N-BEATS, Ensemble)

### Slow Training
- Use TCN or LSTM instead of N-BEATS
- Reduce epochs (try 30 instead of 50)
- Ensure GPU is enabled (check CUDA availability)

### Poor Accuracy
- Increase epochs (try 100 instead of 50)
- Try different models (Ensemble usually best)
- Increase sequence length (try 90 instead of 60)
- Add more training data

## Pro Tips

1. **Start with LSTM** - Get a baseline quickly
2. **Try Attention-LSTM** - Often best balance of speed/accuracy
3. **Use Ensemble for production** - Most robust predictions
4. **Monitor GPU usage** - Check for memory issues
5. **Save your models** - Models auto-save after training
6. **Compare metrics** - Train multiple models and compare R²/RMSE

## Example Workflow

```
Day 1: Quick Baseline
├── Train LSTM (3 min)
└── Evaluate R² = 0.80

Day 2: Try Advanced Models
├── Train Attention-LSTM (4 min) → R² = 0.82
├── Train TCN (3 min) → R² = 0.81
└── Train Transformer (5 min) → R² = 0.83

Day 3: Production Model
├── Train Ensemble (12 min) → R² = 0.85
└── Deploy best model

NEW: All-in-One Comparison 🔥
├── Enable "Train All Models & Compare"
├── Click "Start Training All Models" (~25 min)
├── View comparison table & charts
└── Get automatic recommendation for best model
```

## Command Line Usage (Optional)

### Train Individual Models
```python
# LSTM
from src.model import LSTMModel
from src.trainer import ModelTrainer

model = LSTMModel(seq_length=60, n_features=11, lstm_units=50)
model.build_model()
trainer = ModelTrainer(model)
history = trainer.train(X_train, y_train, epochs=50, batch_size=32)

# Attention-LSTM
from src.attention_lstm_model import AttentionLSTMModel

model = AttentionLSTMModel(seq_length=60, n_features=11, lstm_units=64)
model.build_model()
trainer = ModelTrainer(model)
history = trainer.train(X_train, y_train, epochs=50, batch_size=32)

# N-BEATS
from src.nbeats_model import NBeatsModel
from src.nbeats_trainer import NBeatsTrainer

model = NBeatsModel(seq_length=60, n_features=11, forecast_length=1)
model.build_model()
trainer = NBeatsTrainer(model)
history = trainer.train(X_train, y_train, epochs=50, batch_size=32)
```

### Create Ensemble
```python
from src.ensemble_model import EnsembleModel

# Train individual models first
lstm_model = LSTMModel(...)
attn_model = AttentionLSTMModel(...)
tcn_model = TCNModel(...)

# Create ensemble
models = [(lstm_model, lstm_trainer), (attn_model, attn_trainer), (tcn_model, tcn_trainer)]
ensemble = EnsembleModel(models, method='weighted_average')

# Optimize weights
ensemble.optimize_weights(X_val, y_val)

# Make predictions
predictions = ensemble.predict(X_test)
```

## GPU Monitoring

Check GPU usage during training:
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
```

---

**Happy Modeling!** 🚀
