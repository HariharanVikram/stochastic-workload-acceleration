# Stochastic Workload Acceleration & Microarchitectural Profiling

This project implements and evaluates the performance scaling of a massive stochastic workload (10,000,000 independent Monte Carlo simulation pathways for financial derivative pricing) across three distinct hardware and software execution paradigms. 

## 📊 Performance Benchmarks (10M Paths)

| Hardware Platform | Software Framework / Toolchain | Execution Time | Acceleration Factor | Microarchitectural Paradigm |
| :--- | :--- | :--- | :--- | :--- |
| **Pure CPU Baseline** | Vanilla Python Loops | `10.1232 seconds` | $1\times$ (Baseline) | Sequential execution, high interpreter overhead |
| **Vectorized CPU** | NumPy (C-Backend SIMD) | `0.7971 seconds` | **$12.7\times$** | Automated vector registers (AVX/SSE) allocation |
| **NVIDIA GPU** | PyCUDA / Custom C++ Kernel | `0.3674 seconds` | **$27.5\times$** | Explicit thread scheduling, SIMT warp execution |
| **Google TPU v5e** | JAX / XLA Graph Compiler | `0.000805 seconds` | **$12,575\times$** | Element-wise matrix transformations, Systolic Array MXU |

## 🛠️ Project Architecture & Key Implementations

### 1. Low-Level CUDA Kernel Development (`/notebooks/stochastic_acceleration_gpu.ipynb`)
* Managed explicit GPU grid/block thread layout dimensions to optimize hardware warp occupancy.
* Utilized hardware-level `cuRAND` device primitives to generate parallel statistical distributions locally on the chip, avoiding expensive Host-to-Device data transmission overhead.
* Formulated global memory reads/writes to reduce overall latency bottlenecks.

### 2. XLA Compiler Optimization (`/notebooks/stochastic_acceleration_tpu.ipynb`)
* Refactored the continuous stochastic formula from independent threaded blocks into high-dimension vectorized matrix structures using JAX.
* Leveraged Accelerated Linear Algebra (XLA) graph compilation to fuse operations, completely bypassing traditional thread-scheduling boundaries and running natively on TPU Systolic Array Matrix Multiply Units (MXUs).

## 🚀 How to Run the Notebooks
All components are built to run seamlessly inside Google Colab using free cloud hardware instances. 
1. Open any notebook within the `notebooks/` folder.
2. Select the corresponding runtime accelerator (None for CPU, T4 for GPU, TPU v5e for TPU).
3. Execute the cells sequentially to reproduce the benchmark profiles.
