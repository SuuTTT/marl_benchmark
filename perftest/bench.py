import torch
import time

def benchmark():
    if not torch.cuda.is_available():
        print("CUDA not found!")
        return

    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    print(f"\n🚀 Benchmarking: {name}")
    print("-" * 40)

    # Size for matrix math (8192 x 8192)
    size = 8192
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # 1. FP32 Test (Standard Math)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / 100
    print(f"FP32 Matrix Mul: {fp32_time:.4f} seconds/op")

    # 2. BF16 Test (Modern AI Math - Blackwell's Specialty)
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.matmul(a_bf16, b_bf16)
    torch.cuda.synchronize()
    bf16_time = (time.time() - start) / 100
    print(f"BF16 Matrix Mul: {bf16_time:.4f} seconds/op")

    # 3. Memory Bandwidth Test
    # Copying 1GB of data
    tensor_size = 250 * 1024 * 1024 # ~1GB in float32
    data = torch.randn(tensor_size, device='cpu')
    torch.cuda.synchronize()
    start = time.time()
    _ = data.to(device)
    torch.cuda.synchronize()
    bandwidth = (1.0) / (time.time() - start) # GB/s
    print(f"Host-to-GPU Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    benchmark()