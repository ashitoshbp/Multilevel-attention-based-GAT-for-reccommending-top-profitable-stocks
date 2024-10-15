
This project uses your inbuilt GPU with NVIDIA's CUDA and cuDNN to accelerate deep learning tasks, significantly reducing training time. This README will guide you through the setup and configuration required to enable GPU support and ensure that the library versions are compatible with the code.

## Prerequisites

Before getting started, ensure your system meets the following requirements:

- **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
- **CUDA Toolkit** (version compatible with your GPU)
- **cuDNN** (matching your CUDA version)
- **Python 3.8 or higher**
- **TensorFlow** 

## Installation Guide

### 1. Install NVIDIA CUDA Toolkit

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. Follow the official installation guide to install the CUDA Toolkit: [NVIDIA CUDA Toolkit Installation Guide](https://developer.nvidia.com/cuda-toolkit).

Make sure you download the version of CUDA that matches your GPU architecture. You can verify your GPU architecture using the following command:

```bash
nvidia-smi
```

Once CUDA is installed, add it to your environment path:

```bash
export PATH=/usr/local/cuda-X.X/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Replace `X.X` with the version of CUDA you installed.

### 2. Install cuDNN (CUDA Deep Neural Network Library)

cuDNN is a GPU-accelerated library for deep neural networks, optimized for fast performance on NVIDIA GPUs. Follow these steps:

1. Download cuDNN from [NVIDIA cuDNN Library](https://developer.nvidia.com/cudnn).
2. Extract the downloaded files and move them to the appropriate CUDA directories:

```bash
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

3. Ensure that cuDNN is correctly installed by checking version compatibility:

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### 3. Install Python Libraries

Once CUDA and cuDNN are installed, install the necessary deep learning libraries (TensorFlow or PyTorch) configured to use GPU.

- For TensorFlow:

```bash
pip install tensorflow-gpu
```

- For PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuX.X
```

Replace `X.X` with your CUDA version.

### 4. Verify GPU Configuration

You can verify that TensorFlow or PyTorch is using the GPU by running the following commands:

- **For TensorFlow**:

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```


## Running the Code

Once your environment is set up with CUDA and cuDNN, and the libraries are installed, you can run your code. Make sure to optimize your code to take advantage of the GPU. For example:

- In TensorFlow, your GPU will be used automatically if available.
- In PyTorch, you need to move tensors and models to the GPU explicitly using `.cuda()`:

```python
model = model.cuda()
data = data.cuda()
```

## Version Compatibility

It is crucial to ensure that the versions of CUDA, cuDNN, and your deep learning libraries (TensorFlow, PyTorch, etc.) are compatible. Here are general guidelines for compatibility:

- **CUDA**: The version of CUDA must match the compute capability of your NVIDIA GPU.
- **cuDNN**: The cuDNN version must match the version of CUDA you have installed.
- **TensorFlow** and **PyTorch**: These libraries must be compiled with the appropriate version of CUDA/cuDNN.

