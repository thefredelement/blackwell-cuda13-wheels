# RTX 50x Optimized Python Wheels for CUDA 13.0

High-performance Python wheels optimized for NVIDIA RTX 5070 Ti (Blackwell/sm_120) with CUDA 13.0 support. Get 100 TFLOPs of compute power for ML workloads without waiting for official releases.

## Performance summary

- **100 TFLOPs FP16** with Tensor Cores
- **34 TFLOPs FP32** sustained performance  
- **719 GB/s** memory bandwidth utilization
- **65-131x** speedup on audio processing (~580x realtime length)
- **4.65x** faster attention mechanisms

## Quick start

### Prerequisites
```bash
# NVIDIA Driver 550+ and CUDA 13.0
nvidia-smi  # Should show CUDA 13.0

(You probably need the (at time of writing beta) 580 driver)

# Python 3.12
python3 --version  # Should show 3.12.x
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/rtx5070ti-wheels.git
cd rtx5070ti-wheels

# Install PyTorch ecosystem
pip install torch-2.9.0a0+cudnn9-cp312-cp312-linux_x86_64.whl
pip install torchvision-0.24.0a0+98f8b37-cp312-cp312-linux_x86_64.whl
pip install torchaudio-2.8.0-cp312-cp312-linux_x86_64.whl

# Install scientific computing
pip install numpy-2.3.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
pip install scipy-1.16.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl

# Install ML optimizations
pip install flash_attn-2.8.2-cp312-cp312-linux_x86_64.whl
pip install xformers-0.0.32+635a8ab2.d20250806-cp39-abi3-linux_x86_64.whl
pip install deepspeed-0.17.4+c4b1a8cb-cp312-cp312-linux_x86_64.whl
pip install triton-3.4.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

## Wheels

All wheels are compiled with `-O3 -march=native` optimizations for maximum performance on RTX 50*.

| Package | Version | Key Features | Performance |
|---------|---------|--------------|-------------|
| **PyTorch** | 2.9.0a0+cudnn9 | CUDA 13.0, cuDNN 9.12.0, sm_120 | 34 TFLOPs FP32, 100 TFLOPs FP16 |
| **torchvision** | 0.24.0a0 | Optimized image transforms | 2x speedup with FP16 |
| **torchaudio** | 2.8.0 | GPU-accelerated audio | 65x avg, 131x MFCC |
| **NumPy** | 2.3.2 | MKL optimizations | 1.4 TFLOPs matrix ops |
| **SciPy** | 1.16.1 | Optimized BLAS/LAPACK | Fast linear algebra |
| **Flash Attention** | 2.8.2 | Memory-efficient attention | 4.65x speedup |
| **xformers** | 0.0.32 | Facebook's efficient transformers | 2x+ memory savings |
| **DeepSpeed** | 0.17.4 | Microsoft's training optimization | 3.09x FusedAdam |
| **Triton** | 3.4.0 | OpenAI's GPU compiler | 1.1-2x torch.compile |

WIP: Tflow

## Verify installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")

# Test performance
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
%timeit torch.matmul(x, x)  # Should show ~90 TFLOPs
```

### Measured Performance

| Workload | Performance | Notes |
|----------|-------------|-------|
| Matrix Multiplication (FP32) | 34 TFLOPs | 65% efficiency |
| Matrix Multiplication (FP16) | 100 TFLOPs | Tensor Cores enabled |
| Memory Bandwidth | 719 GB/s | 100% theoretical |
| Conv2D Operations | 2x speedup | With cuDNN 9.12 |
| Attention Mechanisms | 4.65x speedup | Flash Attention |
| Audio Transforms | 65-131x speedup | GPU accelerated |

## Use cases

These wheels are optimized for:
- **Large Language Models**: 7B models at full precision, 13B with quantization
- **Computer Vision**: Real-time inference at 2000+ images/sec
- **Audio Processing**: Real-time processing of 100+ audio streams
- **Embeddings**: Generate embeddings for 10M+ documents/hour
- **Transformers**: Multi-head attention with up to 64 heads
- **Research**: Faster than Colab Pro for most workloads

## Tips

```python
# Enable cuDNN autotuner for best performance
torch.backends.cudnn.benchmark = True

# Use mixed precision for 3x speedup
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)

# Prevent memory fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## System Requirements

- **GPU**: NVIDIA RTX 50* (sm_120 Blackwell GPUs)
- **CUDA**: 13.0
- **Driver**: 580 or newer
- **OS**: Ubuntu 22.04/24.04 (tested on 24.04)
- **Python**: 3.12
- **RAM**: 32GB recommended

## Details

All wheels built with:
- **CUDA**: 13.0
- **cuDNN**: 9.12.0
- **Architecture**: `TORCH_CUDA_ARCH_LIST="12.0+PTX"`
- **Compiler**: GCC 13.3.0
- **Flags**: `-O3 -march=native -mtune=native`
- **Python**: CPython 3.12

## Limitations

- Python 3.12 only (cp312)
- Linux x86_64 only
- Requires RTX 50* or newer Blackwell GPU
- Some wheels are large (PyTorch ~400MB)

## Contributing

Found an issue or want to add more wheels? Please open an issue or PR!

## License

These wheels are compiled from open-source projects. Please refer to each project's original license:
- PyTorch: BSD
- NumPy/SciPy: BSD
- Flash Attention: BSD
- xformers: MIT
- DeepSpeed: MIT
- Triton: MIT

## Acknowledgments

Built on Ubuntu 24.04 with love for the ML community wanting to rock blackwell / sm 120 at home

---

**Note**: These are bleeding-edge builds. Test thoroughly before production use.
