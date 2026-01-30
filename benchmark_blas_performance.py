#!/usr/bin/env python3

import torch
import torch.xpu
import time
import numpy as np

def benchmark_addmm_performance():
    """Benchmark addmm performance: float32 (oneDNN) vs float64 (oneMKL)"""
    print("=" * 70)
    print("ADDMM Performance Benchmark: float32 vs float64")
    print("=" * 70)
    
    if not torch.xpu.is_available():
        print("❌ XPU not available")
        return
    
    # Test different matrix sizes
    sizes = [
        (32, 32, 32),      # Small
        (128, 128, 128),   # Medium  
        (512, 256, 256),   # Large rectangular
        (256, 256, 256),   # Large square
    ]
    
    iterations = 1000
    warmup_iterations = 100
    
    alpha, beta = 2.5, 1.5
    
    for M, K, N in sizes:
        print(f"\nMatrix Size: [{M}x{K}] @ [{K}x{N}] -> [{M}x{N}]")
        print(f"Iterations: {iterations} (after {warmup_iterations} warmup)")
        print("-" * 50)
        
        # Test float32 (oneDNN path)
        print("🔸 Testing float32 (oneDNN path)...")
        
        # Create float32 tensors
        self_f32 = torch.randn(M, N, dtype=torch.float32, device='xpu')
        mat1_f32 = torch.randn(M, K, dtype=torch.float32, device='xpu')
        mat2_f32 = torch.randn(K, N, dtype=torch.float32, device='xpu')
        
        # Warmup
        for _ in range(warmup_iterations):
            torch.addmm(self_f32, mat1_f32, mat2_f32, beta=beta, alpha=alpha)
        
        # Benchmark float32
        torch.xpu.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            result_f32 = torch.addmm(self_f32, mat1_f32, mat2_f32, beta=beta, alpha=alpha)
            
        torch.xpu.synchronize()
        end_time = time.time()
        
        time_f32 = end_time - start_time
        time_per_op_f32 = (time_f32 / iterations) * 1000  # ms
        
        print(f"  Total time: {time_f32:.4f}s")
        print(f"  Time per op: {time_per_op_f32:.4f}ms")
        
        # Test float64 (oneMKL path)  
        print("🔸 Testing float64 (oneMKL path)...")
        
        # Create float64 tensors
        self_f64 = torch.randn(M, N, dtype=torch.float64, device='xpu')
        mat1_f64 = torch.randn(M, K, dtype=torch.float64, device='xpu') 
        mat2_f64 = torch.randn(K, N, dtype=torch.float64, device='xpu')
        
        # Warmup
        for _ in range(warmup_iterations):
            torch.addmm(self_f64, mat1_f64, mat2_f64, beta=beta, alpha=alpha)
            
        # Benchmark float64
        torch.xpu.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            result_f64 = torch.addmm(self_f64, mat1_f64, mat2_f64, beta=beta, alpha=alpha)
            
        torch.xpu.synchronize()
        end_time = time.time()
        
        time_f64 = end_time - start_time
        time_per_op_f64 = (time_f64 / iterations) * 1000  # ms
        
        print(f"  Total time: {time_f64:.4f}s")
        print(f"  Time per op: {time_per_op_f64:.4f}ms")
        
        # Compare
        slowdown_ratio = time_f64 / time_f32
        print(f"\n📊 Comparison:")
        print(f"  float64 vs float32 slowdown: {slowdown_ratio:.2f}x")
        
        if slowdown_ratio > 10:
            print("  🚨 float64 is significantly slower!")
        elif slowdown_ratio > 3:
            print("  ⚠️  float64 is moderately slower")
        elif slowdown_ratio > 1.5:
            print("  📈 float64 is slightly slower (expected)")
        else:
            print("  ✅ float64 performance looks good")

def benchmark_addmv_performance():
    """Benchmark addmv performance: float32 vs float64"""  
    print("\n" + "=" * 70)
    print("ADDMV Performance Benchmark: float32 vs float64")
    print("=" * 70)
    
    if not torch.xpu.is_available():
        print("❌ XPU not available")
        return
        
    # Test different sizes
    sizes = [
        (32, 32),      # Small
        (128, 128),    # Medium
        (512, 256),    # Large rectangular
        (256, 256),    # Large square
    ]
    
    iterations = 1000
    warmup_iterations = 100
    alpha, beta = 2.5, 1.5
    
    for M, K in sizes:
        print(f"\nMatrix-Vector Size: [{M}x{K}] @ [{K}] -> [{M}]")
        print(f"Iterations: {iterations} (after {warmup_iterations} warmup)")
        print("-" * 50)
        
        # Test float32
        print("🔸 Testing float32...")
        
        self_f32 = torch.randn(M, dtype=torch.float32, device='xpu')
        mat_f32 = torch.randn(M, K, dtype=torch.float32, device='xpu') 
        vec_f32 = torch.randn(K, dtype=torch.float32, device='xpu')
        
        # Warmup
        for _ in range(warmup_iterations):
            torch.addmv(self_f32, mat_f32, vec_f32, beta=beta, alpha=alpha)
            
        # Benchmark
        torch.xpu.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            result_f32 = torch.addmv(self_f32, mat_f32, vec_f32, beta=beta, alpha=alpha)
            
        torch.xpu.synchronize()
        end_time = time.time()
        
        time_f32 = end_time - start_time
        time_per_op_f32 = (time_f32 / iterations) * 1000
        
        print(f"  Total time: {time_f32:.4f}s")
        print(f"  Time per op: {time_per_op_f32:.4f}ms")
        
        # Test float64
        print("🔸 Testing float64...")
        
        self_f64 = torch.randn(M, dtype=torch.float64, device='xpu')
        mat_f64 = torch.randn(M, K, dtype=torch.float64, device='xpu')
        vec_f64 = torch.randn(K, dtype=torch.float64, device='xpu')
        
        # Warmup  
        for _ in range(warmup_iterations):
            torch.addmv(self_f64, mat_f64, vec_f64, beta=beta, alpha=alpha)
            
        # Benchmark
        torch.xpu.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            result_f64 = torch.addmv(self_f64, mat_f64, vec_f64, beta=beta, alpha=alpha)
            
        torch.xpu.synchronize()
        end_time = time.time()
        
        time_f64 = end_time - start_time
        time_per_op_f64 = (time_f64 / iterations) * 1000
        
        print(f"  Total time: {time_f64:.4f}s")
        print(f"  Time per op: {time_per_op_f64:.4f}ms")
        
        # Compare
        slowdown_ratio = time_f64 / time_f32  
        print(f"\n📊 Comparison:")
        print(f"  float64 vs float32 slowdown: {slowdown_ratio:.2f}x")
        
        if slowdown_ratio > 10:
            print("  🚨 float64 is significantly slower!")
        elif slowdown_ratio > 3:
            print("  ⚠️  float64 is moderately slower")
        elif slowdown_ratio > 1.5:
            print("  📈 float64 is slightly slower (expected)")
        else:
            print("  ✅ float64 performance looks good")

def test_accuracy():
    """Quick accuracy test to verify implementations work correctly"""
    print("\n" + "=" * 70) 
    print("Quick Accuracy Verification")
    print("=" * 70)
    
    if not torch.xpu.is_available():
        return
        
    M, K, N = 64, 64, 64
    alpha, beta = 2.0, 1.5
    
    # Test addmm accuracy
    print("🔍 Testing addmm accuracy...")
    
    self_cpu = torch.randn(M, N, dtype=torch.float64)
    mat1_cpu = torch.randn(M, K, dtype=torch.float64)
    mat2_cpu = torch.randn(K, N, dtype=torch.float64)
    
    self_xpu = self_cpu.to('xpu')
    mat1_xpu = mat1_cpu.to('xpu')
    mat2_xpu = mat2_cpu.to('xpu')
    
    result_cpu = torch.addmm(self_cpu, mat1_cpu, mat2_cpu, beta=beta, alpha=alpha)
    result_xpu = torch.addmm(self_xpu, mat1_xpu, mat2_xpu, beta=beta, alpha=alpha)
    
    error = torch.max(torch.abs(result_cpu - result_xpu.cpu())).item()
    print(f"  addmm max error: {error:.2e}")
    
    if error < 1e-12:
        print("  ✅ addmm accuracy looks good")
    else:
        print("  ❌ addmm has accuracy issues")
        
    # Test addmv accuracy
    print("🔍 Testing addmv accuracy...")
    
    self_cpu_v = torch.randn(M, dtype=torch.float64)
    mat_cpu_v = torch.randn(M, K, dtype=torch.float64)
    vec_cpu_v = torch.randn(K, dtype=torch.float64)
    
    self_xpu_v = self_cpu_v.to('xpu')
    mat_xpu_v = mat_cpu_v.to('xpu') 
    vec_xpu_v = vec_cpu_v.to('xpu')
    
    result_cpu_v = torch.addmv(self_cpu_v, mat_cpu_v, vec_cpu_v, beta=beta, alpha=alpha)
    result_xpu_v = torch.addmv(self_xpu_v, mat_xpu_v, vec_xpu_v, beta=beta, alpha=alpha)
    
    error_v = torch.max(torch.abs(result_cpu_v - result_xpu_v.cpu())).item()
    print(f"  addmv max error: {error_v:.2e}")
    
    if error_v < 1e-12:
        print("  ✅ addmv accuracy looks good")
    else:
        print("  ❌ addmv has accuracy issues")

if __name__ == "__main__":
    print("PyTorch XPU BLAS Performance Benchmark")
    print("Testing oneDNN (float32) vs oneMKL (float64) implementations")
    print()
    
    test_accuracy()
    benchmark_addmm_performance()
    benchmark_addmv_performance()
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)