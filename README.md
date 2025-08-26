# TFLOPS Benchmark

A PyTorch-based benchmarking tool for measuring matrix multiplication performance (TFLOPS - Tera Floating Point Operations Per Second) across different devices and data types.

## Overview

This benchmark measures the computational throughput of matrix multiplication operations on GPUs (CUDA and MPS) using various floating-point precisions. It helps evaluate hardware performance for deep learning workloads.

## Features

- **Multi-device support**: CUDA and MPS (Apple Silicon) GPUs
- **Multiple data types**: float32, float16, and bfloat16 (when supported)
- **Configurable matrix sizes**: Test with different matrix dimensions
- **Performance visualization**: Generates charts showing performance vs matrix size
- **Automatic device detection**: Selects available GPU automatically

## Usage

### Basic Usage

Run the default benchmark sweep across standard matrix sizes (256, 512, 1024, 2048, 4096, 8192):

```bash
uv run tflops_bench.py
```

This will:
- Test all available GPU devices
- Test all supported data types per device
- Generate performance charts saved as `duration_vs_size_combined.png`

### Single Size Test

Test with a specific matrix size:

```bash
uv run tflops_bench.py --size 4096
```

### Specific Device/Dtype

Test a specific device and data type combination:

```bash
uv run tflops_bench.py --device cuda --dtype float16 --size 2048
```

### Command Line Options

- `--size N`: Matrix size for NxN multiplication (default: runs sweep)
- `--iterations N`: Number of iterations for timing (default: 100)
- `--device [cuda|mps|cpu]`: Specific device to test
- `--dtype [float32|float16|bfloat16]`: Specific data type
- `--verbose`: Show detailed per-test output

## How It Works

1. **Matrix Generation**: Creates random NxN matrices on the target device
2. **Warmup Phase**: Performs 10 warmup iterations to stabilize GPU state
3. **Timed Execution**: Runs the specified number of matrix multiplications
4. **TFLOPS Calculation**: Computes performance as `2*N³*iterations / (time * 10¹²)`
5. **Device Synchronization**: Ensures all operations complete before timing

## Performance Metrics

The benchmark calculates TFLOPS based on the formula:
- FLOPs per matmul = 2 × N³ (for NxN matrices)
- TFLOPS = (Total FLOPs) / (Time in seconds × 10¹²)

## Output

### Console Output
- Per-configuration performance results
- Summary table showing TFLOPS and execution time

### Charts (when running default sweep)
Generates a 4-panel chart showing:
1. Duration vs matrix size (linear scale)
2. Duration vs matrix size (log-log scale)
3. TFLOPS vs matrix size (linear scale)
4. TFLOPS vs matrix size (semi-log scale)

## Device Support Notes

### CUDA
- Supports float32, float16
- Supports bfloat16 on Ampere (SM80) and newer GPUs
- Automatic capability detection

### MPS (Apple Silicon)
- Supports float32, float16
- bfloat16 support depends on macOS version and hardware
- Gracefully handles unsupported configurations

## Requirements

- Python e 3.10
- PyTorch e 2.8.0
- matplotlib e 3.10.5 (for charts)
- numpy e 2.3.2

## Example Results

Typical performance ranges (vary by hardware):
- **Consumer GPUs**: 10-50 TFLOPS (float32), 20-100 TFLOPS (float16)
- **Data center GPUs**: 50-200 TFLOPS (float32), 100-500 TFLOPS (float16/bfloat16)
- **Apple Silicon**: 5-20 TFLOPS depending on chip and precision