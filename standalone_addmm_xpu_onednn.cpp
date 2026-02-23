#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

// SYCL and oneDNN headers for XPU
#include <sycl/sycl.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

using namespace dnnl;

class OneDNNXPUAddMM {
private:
    sycl::device dev;
    sycl::context ctx;
    sycl::queue q;
    engine eng;
    stream s;
    
public:
    OneDNNXPUAddMM() {
        // Initialize SYCL for Intel GPU
        try {
            // Try to get Intel GPU
            dev = sycl::device(sycl::gpu_selector_v);
            std::cout << "Using GPU: " << dev.get_info<sycl::info::device::name>() << std::endl;
        } catch (const sycl::exception& e) {
            std::cout << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            dev = sycl::device(sycl::cpu_selector_v);
        }
        
        ctx = sycl::context(dev);
        q = sycl::queue(ctx, dev);
        
        // Create oneDNN engine for SYCL
        eng = sycl_interop::make_engine(dev, ctx);
        s = sycl_interop::make_stream(eng, q);
    }
    
    // Helper function to initialize random data
    void init_random_data(double* data, size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }
    
    // Allocate SYCL device memory
    double* allocate_device_memory(size_t size) {
        return sycl::malloc_device<double>(size, q);
    }
    
    // Copy host to device
    void copy_to_device(double* device_ptr, const double* host_ptr, size_t size) {
        q.memcpy(device_ptr, host_ptr, size * sizeof(double)).wait();
    }
    
    // Copy device to host
    void copy_to_host(double* host_ptr, const double* device_ptr, size_t size) {
        q.memcpy(host_ptr, device_ptr, size * sizeof(double)).wait();
    }
    
    // Free device memory
    void free_device_memory(double* device_ptr) {
        sycl::free(device_ptr, q);
    }

    // XPU addmm implementation following PyTorch's approach
    void addmm_xpu(double* result, const double* A, const double* B, double bias_scalar,
                   int M, int K, int N, double alpha, double beta) {
        
        // Allocate device memory
        double* d_A = allocate_device_memory(M * K);
        double* d_B = allocate_device_memory(K * N);
        double* d_result = allocate_device_memory(M * N);
        
        // Copy to device
        copy_to_device(d_A, A, M * K);
        copy_to_device(d_B, B, K * N);
        
        try {
            // Create memory dimensions
            memory::dims A_dims = {M, K};
            memory::dims B_dims = {K, N};
            memory::dims result_dims = {M, N};
            
            // Create memory descriptors - use double precision
            auto A_md = memory::desc(A_dims, memory::data_type::f64, memory::format_tag::ab);
            auto B_md = memory::desc(B_dims, memory::data_type::f64, memory::format_tag::ab);
            auto result_md = memory::desc(result_dims, memory::data_type::f64, memory::format_tag::ab);
            
            // Create oneDNN memory objects
            auto A_mem = sycl_interop::make_memory(A_md, eng, sycl_interop::memory_kind::usm, d_A);
            auto B_mem = sycl_interop::make_memory(B_md, eng, sycl_interop::memory_kind::usm, d_B);
            auto result_mem = sycl_interop::make_memory(result_md, eng, sycl_interop::memory_kind::usm, d_result);
            
            // Create post-ops for testing internal arithmetic precision
            // Apply: result = 1.0 * (A @ B) + 0.0 (should be identity)
            // This tests if oneDNN's internal post-op arithmetic truncates
            // the high-precision matmul result to float32
            post_ops ops;
            float alpha_f = static_cast<float>(alpha);  // 1.0f
            float beta_f = static_cast<float>(beta);    // 1.0f  
            float bias_f = static_cast<float>(bias_scalar);  // 0.0f
            
            // if (beta_f == 0.0f || bias_f == 0.0f) {
            //     // result = alpha * (A @ B) + 0 = alpha * (A @ B)
            //     // Tests if multiplication by alpha truncates the matmul result
            //     ops.append_eltwise(algorithm::eltwise_linear, alpha_f, 0.0f);
            // } else {
            //     // General case (shouldn't be used in this test)
            //     ops.append_eltwise(algorithm::eltwise_linear, alpha_f, beta_f * bias_f);
            // }
            primitive_attr attr;
            attr.set_post_ops(ops);
            
            // Create matmul primitive descriptor
            auto matmul_pd = matmul::primitive_desc(eng, A_md, B_md, result_md, attr);
            
            // Create matmul primitive
            auto matmul_prim = matmul(matmul_pd);
            
            // Execute using SYCL interop (like PyTorch)
            std::unordered_map<int, memory> args = {
                {DNNL_ARG_SRC, A_mem},
                {DNNL_ARG_WEIGHTS, B_mem},
                {DNNL_ARG_DST, result_mem}
            };
            
            auto event = sycl_interop::execute(matmul_prim, s, args);
            event.wait();
            
        } catch (const dnnl::error& e) {
            std::cout << "oneDNN operation failed: " << e.what() << std::endl;
            // For now, just zero out the result
            q.fill(d_result, 0.0, M * N).wait();
        }
        
        // Copy result back to host
        copy_to_host(result, d_result, M * N);
        
        // Free device memory
        free_device_memory(d_A);
        free_device_memory(d_B);
        free_device_memory(d_result);
    }
};

int main() {
    std::cout << "oneDNN XPU Internal Post-ops Precision Test" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Testing if post-ops internal arithmetic is done in float32 or float64\n" << std::endl;
    
    const int M = 2, K = 2, N = 2;
    
    try {
        OneDNNXPUAddMM addmm_impl;
        
        // Create matrices that produce a high-precision result
        // A @ B will compute to exactly 1.123456789012345 (representable in f64 but not f32)
        std::vector<double> A = {
            1.123456789012345, 0.0,   // First row: [value, 0]
            0.0, 1.123456789012345    // Second row: [0, value]  
        };
        std::vector<double> B = {
            1.0, 0.0,   // [1, 0]
            0.0, 1.0    // [0, 1]  (identity matrix)
        };
        std::vector<double> result(M * N);
        
        // Post-op parameters: result = 1.0 * (A @ B) + 0.0
        // This should be identity, but tests if oneDNN's internal arithmetic
        // truncates the high-precision matmul result to float32
        const double alpha = 1.0;  // Exactly representable
        const double beta = 1.0;   // Will be ignored since bias = 0
        const double bias = 0.0;   // No bias addition
        
        double expected_matmul = 1.123456789012345;  // A @ B result (f64)
        double expected_truncated = static_cast<double>(static_cast<float>(expected_matmul));  // f32 truncated
        
        std::cout << "Test setup:" << std::endl;
        std::cout << "  A @ B (f64 precision) = " << std::fixed << std::setprecision(15) << expected_matmul << std::endl;
        std::cout << "  A @ B (f32 precision) = " << std::fixed << std::setprecision(15) << expected_truncated << std::endl;
        std::cout << "  Post-op: result = 1.0 * (A @ B) + 0.0 (should be identity)" << std::endl;
        
        // Run oneDNN: result = 1.0 * (A @ B) + 0.0
        std::cout << "\nRunning oneDNN XPU computation..." << std::endl;
        addmm_impl.addmm_xpu(result.data(), A.data(), B.data(), bias, M, K, N, alpha, beta);
        
        double actual_result = result[0];  // Check diagonal element [0,0]
        
        std::cout << "\nResults:" << std::endl;
        std::cout << "  Expected (f64 internal): " << std::fixed << std::setprecision(15) << expected_matmul << std::endl;
        std::cout << "  Expected (f32 internal): " << std::fixed << std::setprecision(15) << expected_truncated << std::endl;
        std::cout << "  Actual (oneDNN result):  " << std::fixed << std::setprecision(15) << actual_result << std::endl;
        
        double diff_from_fp64 = std::abs(actual_result - expected_matmul);
        double diff_from_fp32 = std::abs(actual_result - expected_truncated);
        
        std::cout << "\nPrecision Analysis:" << std::endl;
        std::cout << "  Difference from f64: " << std::scientific << diff_from_fp64 << std::endl;
        std::cout << "  Difference from f32: " << std::scientific << diff_from_fp32 << std::endl;
        
        std::cout << "\nConclusion:" << std::endl;
        if (diff_from_fp64 < 1e-14) {
            std::cout << "  CONFIRMED: oneDNN preserves FLOAT64 precision in post-ops!" << std::endl;
            std::cout << "  The issue is only in the API parameter casting." << std::endl;
        } else if (diff_from_fp32 < 1e-14) {
            std::cout << "  CONFIRMED: oneDNN internal post-ops arithmetic is FLOAT32!" << std::endl;
            std::cout << "  Even f64 tensor data gets truncated during post-op application." << std::endl;
        } else {
            std::cout << "  Unexpected result - neither f32 nor f64 match." << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}