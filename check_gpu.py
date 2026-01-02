"""
GPU Verification Script for RTX 5070
Check if CUDA and GPU are properly configured
"""

import sys

print("=" * 60)
print("GPU & CUDA Configuration Check")
print("=" * 60)

# Check TensorFlow GPU
print("\n1. TensorFlow GPU Check:")
print("-" * 60)
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ Number of GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Get GPU details
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if 'device_name' in gpu_details:
                    print(f"    Device Name: {gpu_details['device_name']}")
                if 'compute_capability' in gpu_details:
                    print(f"    Compute Capability: {gpu_details['compute_capability']}")
            except:
                pass
        
        # Check if TensorFlow can use GPU
        print("\n  Testing GPU computation:")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"  ✓ GPU computation successful: {c.numpy()}")
    else:
        print("✗ No GPU devices found by TensorFlow")
        print("  TensorFlow is running on CPU only")
except ImportError:
    print("✗ TensorFlow not installed")
except Exception as e:
    print(f"✗ Error checking TensorFlow GPU: {e}")

# Check PyTorch GPU
print("\n2. PyTorch GPU Check:")
print("-" * 60)
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"    Name: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Test PyTorch GPU computation
        print("\n  Testing GPU computation:")
        device = torch.device('cuda:0')
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.matmul(x, y)
        print(f"  ✓ GPU computation successful on {device}")
        print(f"  Current device: {torch.cuda.current_device()}")
    else:
        print("✗ CUDA not available for PyTorch")
        print("  PyTorch is running on CPU only")
except ImportError:
    print("✗ PyTorch not installed")
except Exception as e:
    print(f"✗ Error checking PyTorch GPU: {e}")

# Check CUDA environment variables
print("\n3. CUDA Environment Variables:")
print("-" * 60)
import os
cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH']
for var in cuda_vars:
    value = os.environ.get(var, 'Not set')
    print(f"  {var}: {value}")

# System Info
print("\n4. System Information:")
print("-" * 60)
print(f"  Python version: {sys.version}")
print(f"  Platform: {sys.platform}")

# Memory check
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  System RAM: {mem.total / 1024**3:.2f} GB")
    print(f"  Available RAM: {mem.available / 1024**3:.2f} GB")
except:
    pass

print("\n" + "=" * 60)
print("Recommendations:")
print("=" * 60)

try:
    import tensorflow as tf
    import torch
    
    tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    torch_gpu = torch.cuda.is_available()
    
    if tf_gpu and torch_gpu:
        print("✓ Both TensorFlow and PyTorch have GPU access!")
        print("✓ Your RTX 5070 is ready for training.")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Select either LSTM or N-BEATS model")
        print("3. Train with GPU acceleration enabled")
    elif tf_gpu and not torch_gpu:
        print("⚠ TensorFlow has GPU access, but PyTorch doesn't")
        print("  LSTM will use GPU, N-BEATS will use CPU")
        print("\nTo fix PyTorch:")
        print("  pip uninstall torch torchvision torchaudio")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    elif not tf_gpu and torch_gpu:
        print("⚠ PyTorch has GPU access, but TensorFlow doesn't")
        print("  N-BEATS will use GPU, LSTM will use CPU")
        print("\nTo fix TensorFlow:")
        print("  pip install tensorflow[and-cuda]>=2.15.0")
    else:
        print("✗ Neither framework can access the GPU")
        print("\nTroubleshooting steps:")
        print("1. Verify CUDA is installed: nvidia-smi")
        print("2. Check CUDA version compatibility")
        print("3. Reinstall GPU packages:")
        print("   pip install -r requirements.txt --force-reinstall")
except:
    print("Install required packages first:")
    print("  pip install -r requirements.txt")

print("=" * 60)
