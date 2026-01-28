#!/usr/bin/env python3

import torch
import time
import statistics
import json
import argparse
import os
from datetime import datetime

def benchmark_fp32_xpu(output_file=None):
    """Benchmark FP32 performance on XPU only"""
    print("🔍 FP32 Performance Analysis on XPU")
    print("=" * 60)
    
    if not torch.xpu.is_available():
        print("❌ XPU not available")
        return None, None
    
    # Test scenarios with different matrix sizes
    test_configs = [
        # (name, B, M, K, N, iterations)
        ("Very Tiny", 10, 3, 3, 3, 15000),
        ("Tiny", 10, 16, 16, 16, 10000),
        ("Small", 10, 64, 64, 64, 10000),
        ("Medium", 10, 256, 256, 256, 10000),
        ("Large", 10, 512, 512, 512, 10000),
        ("X-Large", 10, 1024, 1024, 1024, 1000),
        ("XX-Large", 5, 2048, 2048, 2048, 200),
        ("XXX-Large", 5, 4096, 4096, 4096, 20),
    ]
    
    num_runs = 5  # Number of times to run each test for averaging
    all_addmm_results = []
    all_baddbmm_results = []
    
    print(f"Running {num_runs} iterations per test for statistical accuracy...\n")
    
    for name, B, M, K, N, iterations in test_configs:
        # ADDMM test (2D matrices)
        print(f"🧪 Testing ADDMM {name} matrices ({M}×{K} @ {K}×{N}, {M*K*N:,} elements)")
        
        # Storage for multiple runs
        xpu_fp32_times = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)
            
            # XPU FP32 ADDMM
            a_xpu_fp32 = torch.randn(M, K, dtype=torch.float32, device='xpu')
            b_xpu_fp32 = torch.randn(K, N, dtype=torch.float32, device='xpu')
            c_xpu_fp32 = torch.randn(M, N, dtype=torch.float32, device='xpu')
            
            # Warmup
            for _ in range(5):
                torch.addmm(c_xpu_fp32, a_xpu_fp32, b_xpu_fp32)
            torch.xpu.synchronize()
            
            start_time = time.time()
            for _ in range(iterations):
                torch.addmm(c_xpu_fp32, a_xpu_fp32, b_xpu_fp32)
            torch.xpu.synchronize()
            xpu_fp32_time = time.time() - start_time
            xpu_fp32_times.append(xpu_fp32_time)
            
            print("✓")
        
        # Calculate ADDMM statistics
        xpu_fp32_mean = statistics.mean(xpu_fp32_times)
        xpu_fp32_std = statistics.stdev(xpu_fp32_times) if len(xpu_fp32_times) > 1 else 0
        
        print(f"    ADDMM XPU FP32: {xpu_fp32_mean:.4f}±{xpu_fp32_std:.4f}s")
        print()
        
        all_addmm_results.append({
            'name': name,
            'size': M * K * N,
            'B': B, 'M': M, 'K': K, 'N': N,
            'iterations': iterations,
            'xpu_fp32_mean': xpu_fp32_mean,
            'xpu_fp32_std': xpu_fp32_std,
            'xpu_fp32_times': xpu_fp32_times,
            'op': 'addmm'
        })
        
        # BADDBMM test (3D batch matrices)
        print(f"🧪 Testing BADDBMM {name} batch matrices ({B}×{M}×{K} @ {B}×{K}×{N}, {B*M*K*N:,} elements)")
        
        # Storage for multiple runs
        xpu_fp32_times = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)
            
            # XPU FP32 BADDBMM
            input_fp32 = torch.randn(B, M, N, dtype=torch.float32, device='xpu')
            batch1_fp32 = torch.randn(B, M, K, dtype=torch.float32, device='xpu')
            batch2_fp32 = torch.randn(B, K, N, dtype=torch.float32, device='xpu')
            
            # Warmup
            for _ in range(5):
                torch.baddbmm(input_fp32, batch1_fp32, batch2_fp32)
            torch.xpu.synchronize()
            
            start_time = time.time()
            for _ in range(iterations):
                torch.baddbmm(input_fp32, batch1_fp32, batch2_fp32)
            torch.xpu.synchronize()
            xpu_fp32_time = time.time() - start_time
            xpu_fp32_times.append(xpu_fp32_time)
            
            print("✓")
        
        # Calculate BADDBMM statistics
        xpu_fp32_mean = statistics.mean(xpu_fp32_times)
        xpu_fp32_std = statistics.stdev(xpu_fp32_times) if len(xpu_fp32_times) > 1 else 0
        
        print(f"    BADDBMM XPU FP32: {xpu_fp32_mean:.4f}±{xpu_fp32_std:.4f}s")
        print("-" * 60)
        
        all_baddbmm_results.append({
            'name': name,
            'size': B * M * K * N,
            'B': B, 'M': M, 'K': K, 'N': N,
            'iterations': iterations,
            'xpu_fp32_mean': xpu_fp32_mean,
            'xpu_fp32_std': xpu_fp32_std,
            'xpu_fp32_times': xpu_fp32_times,
            'op': 'baddbmm'
        })
    
    # Save results to file if specified
    if output_file:
        save_results(all_addmm_results, all_baddbmm_results, output_file)
    
    return all_addmm_results, all_baddbmm_results

def save_results(addmm_results, baddbmm_results, output_file):
    """Save benchmark results to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'device_name': torch.xpu.get_device_name() if torch.xpu.is_available() else "Unknown",
        'addmm_results': addmm_results,
        'baddbmm_results': baddbmm_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"📁 Results saved to: {output_file}")

def load_results(input_file):
    """Load benchmark results from JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def compare_results(file1, file2):
    """Compare results from two benchmark files"""
    print("🔍 COMPARING BENCHMARK RESULTS")
    print("=" * 80)
    
    data1 = load_results(file1)
    data2 = load_results(file2)
    
    print(f"📊 File 1: {file1}")
    print(f"   Timestamp: {data1['timestamp']}")
    print(f"   PyTorch: {data1['pytorch_version']}")
    print(f"   Device: {data1['device_name']}")
    print()
    
    print(f"📊 File 2: {file2}")
    print(f"   Timestamp: {data2['timestamp']}")
    print(f"   PyTorch: {data2['pytorch_version']}")
    print(f"   Device: {data2['device_name']}")
    print()
    
    # Compare ADDMM results
    print("=" * 80)
    print("📈 ADDMM PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Size':<20} {'File1 Time':<12} {'File2 Time':<12} {'Speedup':<10} {'Change'}")
    print("-" * 80)
    
    for r1, r2 in zip(data1['addmm_results'], data2['addmm_results']):
        if r1['name'] == r2['name']:
            time1 = r1['xpu_fp32_mean']
            time2 = r2['xpu_fp32_mean']
            speedup = time1 / time2
            change_pct = ((time2 - time1) / time1) * 100
            
            change_str = f"{change_pct:+.1f}%"
            if change_pct < -5:
                change_str += " 🟢 (faster)"
            elif change_pct > 5:
                change_str += " 🔴 (slower)"
            else:
                change_str += " 🟡 (similar)"
            
            print(f"{r1['name']:<20} {time1:>11.4f}s "
                  f"{time2:>11.4f}s "
                  f"{speedup:>9.2f}x "
                  f"{change_str}")
    
    # Compare BADDBMM results
    print("\n" + "=" * 80)
    print("📈 BADDBMM PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Size':<20} {'File1 Time':<12} {'File2 Time':<12} {'Speedup':<10} {'Change'}")
    print("-" * 80)
    
    for r1, r2 in zip(data1['baddbmm_results'], data2['baddbmm_results']):
        if r1['name'] == r2['name']:
            time1 = r1['xpu_fp32_mean']
            time2 = r2['xpu_fp32_mean']
            speedup = time1 / time2
            change_pct = ((time2 - time1) / time1) * 100
            
            change_str = f"{change_pct:+.1f}%"
            if change_pct < -5:
                change_str += " 🟢 (faster)"
            elif change_pct > 5:
                change_str += " 🔴 (slower)"
            else:
                change_str += " 🟡 (similar)"
            
            print(f"{r1['name']:<20} {time1:>11.4f}s "
                  f"{time2:>11.4f}s "
                  f"{speedup:>9.2f}x "
                  f"{change_str}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("🎯 SUMMARY STATISTICS")
    print("=" * 80)
    
    # Calculate overall performance changes
    addmm_speedups = []
    baddbmm_speedups = []
    
    for r1, r2 in zip(data1['addmm_results'], data2['addmm_results']):
        if r1['name'] == r2['name']:
            speedup = r1['xpu_fp32_mean'] / r2['xpu_fp32_mean']
            addmm_speedups.append(speedup)
    
    for r1, r2 in zip(data1['baddbmm_results'], data2['baddbmm_results']):
        if r1['name'] == r2['name']:
            speedup = r1['xpu_fp32_mean'] / r2['xpu_fp32_mean']
            baddbmm_speedups.append(speedup)
    
    if addmm_speedups:
        addmm_avg_speedup = statistics.mean(addmm_speedups)
        print(f"ADDMM Average Speedup: {addmm_avg_speedup:.2f}x")
        
    if baddbmm_speedups:
        baddbmm_avg_speedup = statistics.mean(baddbmm_speedups)
        print(f"BADDBMM Average Speedup: {baddbmm_avg_speedup:.2f}x")
    
    if addmm_speedups and baddbmm_speedups:
        overall_speedup = statistics.mean(addmm_speedups + baddbmm_speedups)
        print(f"Overall Average Speedup: {overall_speedup:.2f}x")

def print_single_summary_table(addmm_results, baddbmm_results):
    """Print formatted summary table for single run"""
    print("=" * 60)
    print("📊 ADDMM PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Size':<20} {'Elements':<20} {'FP32 Time':<12}")
    print("-" * 60)
    
    for result in addmm_results:
        print(f"{result['name']:<20} {result['size']:>20,} "
              f"{result['xpu_fp32_mean']:>11.4f}s")
    
    print("\n" + "=" * 60)
    print("📊 BADDBMM PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Size':<20} {'Elements':<20} {'FP32 Time':<12}")
    print("-" * 60)
    
    for result in baddbmm_results:
        print(f"{result['name']:<20} {result['size']:>20,} "
              f"{result['xpu_fp32_mean']:>11.4f}s")

def main():
    """Main execution with argument parsing"""
    parser = argparse.ArgumentParser(description='FP32 XPU Benchmark Tool')
    parser.add_argument('--output', '-o', help='Output file for benchmark results')
    parser.add_argument('--compare', '-c', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare results from two benchmark files')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode
        file1, file2 = args.compare
        if not os.path.exists(file1):
            print(f"❌ File not found: {file1}")
            return
        if not os.path.exists(file2):
            print(f"❌ File not found: {file2}")
            return
        
        compare_results(file1, file2)
    else:
        # Benchmark mode
        print("🚀 Starting FP32 XPU Performance Benchmark")
        print(f"PyTorch version: {torch.__version__}")
        if torch.xpu.is_available():
            print(f"XPU device: {torch.xpu.get_device_name()}")
        print()
        
        output_file = args.output
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_fp32_{timestamp}.json"
        
        addmm_results, baddbmm_results = benchmark_fp32_xpu(output_file)
        
        if addmm_results and baddbmm_results:
            print_single_summary_table(addmm_results, baddbmm_results)
            
            print("\n" + "=" * 60)
            print("✅ Benchmark completed successfully!")
            print(f"📊 Results saved to: {output_file}")
            print(f"💡 To compare with another run, use:")
            print(f"   python {__file__} --compare {output_file} <other_file.json>")

if __name__ == "__main__":
    main()
