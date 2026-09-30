# Lab 01

## Goal

Set up Python virtual environments, install PyTorch and the CUDA toolkit, then use a simple PyTorch program that we inspect with the Nsight Systems profiler to see the GPU functions called and their execution times.

## Create a Python virtual environment

Download the `virtualenv` package:

```bash
wget https://bootstrap.pypa.io/virtualenv/3.10/virtualenv.pyz
```

Create a virtual environment:

```bash
python3 virtualenv.pyz .venv
```

Check the GPU, driver and CUDA execution environment version:

```bash
nvidia-smi
```

## Install the CUDA toolkit

CUDA Toolkit 12.1 should be compatible with the driver:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run

mkdir -p "$HOME/.local/cuda-12.1"

sh cuda_12.1.1_530.30.02_linux.run \
  --silent \
  --toolkit \
  --toolkitpath="$HOME/.local/cuda-12.1"
```

## Install PyTorch

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install a PyTorch version compatible with the CUDA toolkit. For example, `cu121` means that PyTorch was compiled with CUDA Toolkit 12.1:

```bash
pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

## Create an activation script

Create the file:

```bash
touch activate.sh
```

Add the following content:

```bash
#!/bin/bash

# Activate the virtual environment.
source .venv/bin/activate

# Set CUDA paths.
export CUDA_HOME="$HOME/.local/cuda-12.1"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Set up the environment by sourcing the script:

```bash
source activate.sh
```

Check whether the CUDA toolkit installation was successful:

```bash
nvcc --version
```

Check whether PyTorch sees the GPU:

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

## Create a PyTorch test

Create a test script in the `torch_test` directory:

```bash
mkdir torch_test
cd torch_test
touch torch_test.py
```

Add the following content to `torch_test.py`:

```python
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

a = torch.randn(8192, 8192, device="cuda")
b = torch.randn(8192, 8192, device="cuda")

# Warmup
for _ in range(3):
    c = a @ b

torch.cuda.synchronize()

for _ in range(10):
    c = a @ b

torch.cuda.synchronize()

print(c[0, 0].item())
```

The script multiplies two 8192 by 8192 matrices 3+10 times. Run the test:

```bash
python3 torch_test.py
```

## Profile the test with Nsight Systems

Profile the script to see what code was executed on the GPU:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  -o torch_profile \
  python torch_test.py
```

The command creates a report in `torch_profile.nsys-rep`. Open it in the profiler application:

```bash
nsys-ui torch_profile.nsys-rep
```

Now we can investigate which GPU kernels are called by which CPU thread.

## Exercise

We see many kernel calls and want to check the interesting ones. Wrap the kernel calls we are interested in with NVTX ranges and filter them in Nsight Systems:

```python
with torch.cuda.nvtx.range("layer_3"):
    # Computation of the third layer of the neural network.
    pass

with torch.cuda.nvtx.range("mlp"):
    # A kernel call that computes the output of the multilayer perceptron.
    pass
```
