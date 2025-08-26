#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.8.0",
#     "matplotlib>=3.10.5",
#     "numpy>=2.3.2",
# ]
# ///

import torch
import time
import argparse

# Default matrix sizes for the benchmark sweep
DEFAULT_SIZES = [256, 512, 1024, 2048, 4096, 8192]


def cuda_supports_bfloat16() -> bool:
    """Return True if the current CUDA device supports bfloat16.

    Prefer torch.cuda.is_bf16_supported when available; otherwise fall back
    to compute capability (Ampere/SM80 or newer).
    """
    if not torch.cuda.is_available():
        return False
    # Prefer built-in capability check when available
    try:
        supported = torch.cuda.is_bf16_supported()  # type: ignore[attr-defined]
        if isinstance(supported, bool):
            return supported
    except Exception:
        pass
    # Fallback to compute capability: Ampere (SM80) and newer support bf16
    try:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def measure_tflops(size=2048, iterations=100, device_type='auto', dtype_str='bfloat16', verbose: bool = False):
    # Validate and set device (GPU-only by default)
    if device_type == 'auto' or device_type is None:
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            print("No GPU available (CUDA/MPS). Skipping.")
            return None
    
    if device_type == 'cuda':
        if not torch.cuda.is_available():
            print(f"Skipping CUDA - device not available")
            return None
        device = torch.device('cuda')
    elif device_type == 'mps':
        if not torch.backends.mps.is_available():
            print(f"Skipping MPS - device not available")
            return None
        device = torch.device('mps')
    elif device_type == 'cpu':
        # CPU path intentionally supported only when explicitly requested
        device = torch.device('cpu')
    else:
        print(f"Invalid device type: {device_type}. Use 'cuda' or 'mps' (CPU only when explicitly requested).")
        return None
    
    # Set dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    if dtype_str not in dtype_map:
        print(f"Invalid dtype: {dtype_str}")
        return None
    
    dtype = dtype_map[dtype_str]
    
    # Check if dtype is supported on device
    # For MPS bfloat16, attempt the operation and surface errors if unsupported
    
    if device_type == 'cpu' and dtype_str == 'float16' and verbose:
        print(f"Note: float16 on CPU may be slower than float32")
    
    if verbose:
        print(f"\nTesting {device_type.upper()} with {dtype_str}:")
        print(f"  Matrix size: {size}x{size}")
    try:
        A = torch.randn(size, size, device=device, dtype=dtype)
        B = torch.randn(size, size, device=device, dtype=dtype)

        # Warmup
        for _ in range(10):
            C = torch.matmul(A, B)

        # Synchronize based on device
        if device_type == 'cuda':
            torch.cuda.synchronize()
        elif device_type == 'mps':
            torch.mps.synchronize()
        
        start = time.perf_counter()

        for _ in range(iterations):
            C = torch.matmul(A, B)

        # Synchronize based on device
        if device_type == 'cuda':
            torch.cuda.synchronize()
        elif device_type == 'mps':
            torch.mps.synchronize()
        
        end = time.perf_counter()

        flops = 2 * size * size * size * iterations
        time_elapsed = end - start
        tflops = flops / (time_elapsed * 1e12)

        if verbose:
            print(f"  Performance: {tflops:.2f} TFLOPS, time: {time_elapsed:.4f}s")
        return (tflops, time_elapsed)
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_all_tests(size=2048, iterations=100, verbose: bool = False):
    # GPU-only test set
    devices = []
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    if not devices:
        print("No GPU devices available (CUDA/MPS). Nothing to run.")
        return {}
    
    results = {}
    
    print("=" * 60)
    print(f"Running TFLOPS benchmark (GPU-only)")
    print(f"Matrix size: {size}x{size}, Iterations: {iterations}")
    print("=" * 60)
    
    for device_type in devices:
        results[device_type] = {}
        # Select dtypes per-device
        if device_type == 'cuda':
            test_dtypes = ['float32', 'float16']
            if cuda_supports_bfloat16():
                test_dtypes.append('bfloat16')
        elif device_type == 'mps':
            # Try bfloat16 on MPS as well; errors will be caught per test
            test_dtypes = ['float32', 'float16', 'bfloat16']
        else:
            test_dtypes = ['float32']
            
        for dtype_str in test_dtypes:
            result = measure_tflops(size=size, iterations=iterations, 
                                    device_type=device_type, dtype_str=dtype_str, verbose=verbose)
            if result is not None:
                tflops, seconds = result
                results[device_type][dtype_str] = {"tflops": tflops, "seconds": seconds}
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for device_type in devices:
        if results[device_type]:
            print(f"\n{device_type.upper()}:")
            for dtype_str, res in results[device_type].items():
                print(f"  {dtype_str:8s}: {res['tflops']:8.2f} TFLOPS in {res['seconds']:.4f}s")
    
    return results


def run_sizes_and_save_chart(sizes=None, iterations=100, verbose: bool = False, outfile: str = "duration_vs_size.png"):
    """Run the benchmark across multiple sizes and save a chart of duration vs size.

    - Runs the existing GPU-only test matrix for each size in `sizes`.
    - Collects per (device, dtype) durations.
    - Saves a chart with size on the x-axis and time (seconds) on the y-axis.
    """
    if sizes is None:
        sizes = DEFAULT_SIZES

    # Map combo key (e.g., "cuda-float16") -> list of (size, seconds)
    series = {}

    for s in sizes:
        size_results = run_all_tests(size=s, iterations=iterations, verbose=verbose)
        if not size_results:
            continue
        for device_type, dtype_map in size_results.items():
            for dtype_str, metrics in dtype_map.items():
                key = f"{device_type}-{dtype_str}"
                series.setdefault(key, []).append((s, metrics["seconds"]))

    if not series:
        print("No results collected; chart will not be generated.")
        return series

    # Lazy import of matplotlib so the script can still run without it
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Could not import matplotlib; skipping chart generation: {e}")
        return series

    plt.figure(figsize=(8, 5))
    for key, points in series.items():
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if len(xs) >= 1:
            plt.plot(xs, ys, marker='o', label=key)

    plt.title(f"Matmul duration vs size (iterations={iterations})")
    plt.xlabel("Matrix size (N)")
    plt.ylabel("Time (s)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(outfile)
        print(f"Saved chart to {outfile}")
    except Exception as e:
        print(f"Failed to save chart '{outfile}': {e}")
    finally:
        plt.close()

    return series


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure TFLOPS using PyTorch matmul on various devices.")
    parser.add_argument("--size", type=int, default=None, help="Matrix size N for NxN matmul. If omitted, runs default size sweep.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of matmuls to time.")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Test all GPU device/dtype combinations (default mode when no device/dtype provided).")
    parser.add_argument("--device", type=str, default=None, help="Specific device to test (cpu, cuda, mps).")
    parser.add_argument("--dtype", type=str, default=None, help="Specific dtype to test (float32, float16, bfloat16).")
    parser.add_argument("--verbose", action="store_true", help="Show per-test details (Testing..., Performance...).")
    args = parser.parse_args()

    # Default to running all GPU tests when no specific device/dtype is given
    if args.device is None and args.dtype is None:
        if args.size is None:
            # Default: run the sweep over preset sizes and save a chart
            run_sizes_and_save_chart(sizes=DEFAULT_SIZES, iterations=args.iterations, verbose=args.verbose)
        else:
            # Single-size run without chart
            run_all_tests(size=args.size, iterations=args.iterations, verbose=args.verbose)
    else:
        # Run specific test
        # Auto selects a GPU (CUDA preferred) if available
        device_type = args.device if args.device else 'auto'
        dtype_str = args.dtype if args.dtype else 'float32'
        # Single test runs are verbose by default unless --verbose is omitted
        size = args.size if args.size is not None else 2048
        measure_tflops(size=size, iterations=args.iterations,
                       device_type=device_type, dtype_str=dtype_str,
                       verbose=True if not args.verbose else args.verbose)
