import torch
import time
import sys
import argparse

def measure_tflops_mps(size=4096, iterations=100):
    if not torch.backends.mps.is_available():
        print("MPS device not available. Ensure PyTorch is built with MPS and running on Apple silicon.")
        sys.exit(1)

    device = torch.device('mps')  # Metal Performance Shaders
    dtype = torch.bfloat16

    A = torch.randn(size, size, device=device, dtype=dtype)
    B = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)

    torch.mps.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        C = torch.matmul(A, B)

    torch.mps.synchronize()
    end = time.perf_counter()

    flops = 2 * size * size * size * iterations
    time_elapsed = end - start
    tflops = flops / (time_elapsed * 1e12)

    print(f"Performance: {tflops:.2f} TFLOPS")
    return tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure TFLOPS on Apple MPS using PyTorch matmul.")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size N for NxN matmul.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of matmuls to time.")
    args = parser.parse_args()

    measure_tflops_mps(size=args.size, iterations=args.iterations)