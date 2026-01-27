#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/xpu/Blas.h>
#include <torch/library.h>
//#if defined(USE_ONEMKL_XPU)
#include <comm/Runtime.h>
#include <oneapi/mkl/blas.hpp>
#include <c10/xpu/XPUFunctions.h>
//#endif
#ifndef AT_PER_OPERATOR_HEADERS

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mm_native.h>
#endif

namespace at::native {
namespace xpu {

static int count_addmm_out = 0;
static int count_addmv_out = 0;
static int count_baddbmm_out = 0;

void print_logs(){
  // std::cout << "addmm: " << count_addmm_out << ", addmv: " << count_addmv_out << ", baddbmm: " << count_baddbmm_out << std::endl;
}

//#if defined(USE_ONEMKL_XPU)
// Helper functions adapted from the existing oneMKL implementation

static inline bool is_column_major_f64(
    const int64_t stride_first,
    const int64_t stride_second,
    const int64_t dim_first,
    const int64_t dim_second,
    const bool contiguous_batch,
    const bool check_dim_second = true) {
  return contiguous_batch && stride_first == 1 &&
      ((dim_second == 1 && check_dim_second) ||
       stride_second >= std::max(int64_t{1}, dim_first));
}

static inline bool is_row_major_f64(
    const int64_t stride_first,
    const int64_t stride_second,
    const int64_t dim_first,
    const int64_t dim_second,
    const bool contiguous_batch,
    const bool check_dim_first = true) {
  return contiguous_batch && stride_second == 1 &&
      ((dim_first == 1 && check_dim_first) ||
       stride_first >= std::max(int64_t{1}, dim_second));
}

static std::pair<Tensor, bool> process_result_matrix_f64(
    const Tensor& result,
    IntArrayRef result_sizes) {
  const auto result_strides = result.strides();
  const int64_t ndim = result_strides.size();
  const int64_t last_dim = ndim - 1;
  const int64_t second_last_dim = ndim - 2;

  const bool contiguous_batch = ndim > 2
      ? result_strides[0] == (result_sizes[1] * result_sizes[2])
      : true;

  Tensor c = result.resolve_conj();

  if (is_column_major_f64(
          result_strides[second_last_dim],
          result_strides[last_dim],
          result_sizes[second_last_dim],
          result_sizes[last_dim],
          contiguous_batch)) {
    return {c, false};
  }

  if (is_row_major_f64(
          result_strides[second_last_dim],
          result_strides[last_dim],
          result_sizes[second_last_dim],
          result_sizes[last_dim],
          contiguous_batch)) {
    return {c, true};
  }

  // Matrix is not contiguous - make it contiguous while preserving layout
  c = c.transpose(second_last_dim, last_dim)
          .contiguous()
          .transpose_(second_last_dim, last_dim);
  return {c, false};
}

static std::pair<Tensor, bool> process_matrix_f64(
    const Tensor& m,
    bool transpose_c,
    int64_t first_dim,
    int64_t second_dim) {
  const auto m_strides = m.strides();
  const int64_t ndim = m_strides.size();
  const int64_t last_stride = m_strides[ndim - 1];
  const int64_t second_last_stride = m_strides[ndim - 2];

  const bool contiguous_batch =
      ndim > 2 ? m_strides[0] == (m.sizes()[1] * m.sizes()[2]) : true;

  const int64_t stride_inner = transpose_c ? last_stride : second_last_stride;
  const int64_t stride_outer = transpose_c ? second_last_stride : last_stride;

  if (is_column_major_f64(
          stride_inner,
          stride_outer,
          first_dim,
          second_dim,
          contiguous_batch,
          false)) {
    return {m.resolve_conj(), false};
  }

  if (is_row_major_f64(
          stride_inner,
          stride_outer,
          first_dim,
          second_dim,
          contiguous_batch,
          false)) {
    return {m, true};
  }

  // Matrix needs to be made contiguous with transposition based on transpose_c
  return {m.clone(MemoryFormat::Contiguous), !transpose_c};
}

static inline int64_t get_ldc_f64(const bool is_transposed, const Tensor& c) {
  int64_t ldc;
  const int64_t ndim = c.dim();

  // Handle the corner case where the last two strides are both 1
  if (c.strides()[ndim - 2] == c.strides()[ndim - 1] &&
      c.strides()[ndim - 1] == 1) {
    ldc = c.sizes()[is_transposed ? ndim - 1 : ndim - 2];
  } else {
    ldc = c.strides()[is_transposed ? ndim - 2 : ndim - 1];
  }
  return ldc;
}

static Tensor prepare_result_tensor_f64(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2) {
  Tensor result = self.contiguous().resolve_conj().clone().detach();

  std::vector<int64_t> expected_output_size{mat1.size(0), mat2.size(1)};

  if (result.sizes() != expected_output_size) {
    result = broadcast_to(result, expected_output_size).contiguous();
  }

  return result;
}

static void copy_result_to_output_f64(Tensor& output, const Tensor& result) {
  if (!output.is_same(result)) {
    if (output.sizes() == result.sizes()) {
      output.copy_(result);
    } else {
      output.copy_(result.view(output.sizes()));
    }
  }
}

// oneMKL implementation for float64 addmm operation
Tensor& addmm_f64_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  
  // Prepare result tensor following the existing pattern
  Tensor result = prepare_result_tensor_f64(self, mat1, mat2);

  const int64_t ndim = mat1.dim();
  const auto result_sizes = result.sizes();
  
  auto [c, transpose_c] = process_result_matrix_f64(result, result_sizes);
  
  // For cases when C matrix is transposed we need to switch m1 and m2 to use
  // column_major implementation.
  const Tensor& m1 = transpose_c ? mat2 : mat1;
  const Tensor& m2 = transpose_c ? mat1 : mat2;

  const int64_t m = result_sizes[transpose_c ? ndim - 1 : ndim - 2];
  const int64_t n = result_sizes[transpose_c ? ndim - 2 : ndim - 1];
  const int64_t k = mat1.sizes()[ndim - 1];

  auto [a, transpose_a] = process_matrix_f64(m1, transpose_c, m, k);
  auto [b, transpose_b] = process_matrix_f64(m2, transpose_c, k, n);

  const int64_t lda =
      a.strides()[(transpose_a == transpose_c) ? ndim - 1 : ndim - 2];
  const int64_t ldb =
      b.strides()[(transpose_b == transpose_c) ? ndim - 1 : ndim - 2];
  const int64_t ldc = get_ldc_f64(transpose_c, c);

  const double* A = a.data_ptr<double>();
  const double* B = b.data_ptr<double>();
  double* C = c.data_ptr<double>();
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  const oneapi::mkl::transpose transA = transpose_a ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;
  const oneapi::mkl::transpose transB = transpose_b ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;

  oneapi::mkl::blas::column_major::gemm(
      queue, transA, transB, m, n, k, 
      alpha.to<double>(), A, lda, B, ldb, 
      beta.to<double>(), C, ldc);

  copy_result_to_output_f64(out, c);
  
  return out;
}

// oneMKL implementation for float64 baddbmm operation (batched)
Tensor& baddbmm_f64_out_xpu_mkl(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  
  const int64_t batch_size = batch1.size(0);
  const int64_t m = batch1.size(1);
  const int64_t k = batch1.size(2);
  const int64_t n = batch2.size(2);
  
  // Prepare result tensor
  Tensor result = input.contiguous().resolve_conj().clone().detach();
  
  std::vector<int64_t> expected_output_size{batch_size, m, n};
  if (result.sizes() != expected_output_size) {
    result = broadcast_to(result, expected_output_size).contiguous();
  }
  
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  
  // Process each batch
  for (int64_t b = 0; b < batch_size; ++b) {
    Tensor batch1_slice = batch1.select(0, b);
    Tensor batch2_slice = batch2.select(0, b);
    Tensor result_slice = result.select(0, b);
    
    const auto result_sizes = result_slice.sizes();
    auto [c, transpose_c] = process_result_matrix_f64(result_slice, result_sizes);
    
    // For cases when C matrix is transposed we need to switch m1 and m2
    const Tensor& m1 = transpose_c ? batch2_slice : batch1_slice;
    const Tensor& m2 = transpose_c ? batch1_slice : batch2_slice;
    
    const int64_t m_dim = transpose_c ? n : m;
    const int64_t n_dim = transpose_c ? m : n;
    
    auto [a, transpose_a] = process_matrix_f64(m1, transpose_c, m_dim, k);
    auto [b_tensor, transpose_b] = process_matrix_f64(m2, transpose_c, k, n_dim);
    
    const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
    const int64_t ldb = b_tensor.strides()[(transpose_b == transpose_c) ? 1 : 0];
    const int64_t ldc = get_ldc_f64(transpose_c, c);
    
    const double* A = a.data_ptr<double>();
    const double* B = b_tensor.data_ptr<double>();
    double* C = c.data_ptr<double>();
    
    const oneapi::mkl::transpose transA = transpose_a ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;
    const oneapi::mkl::transpose transB = transpose_b ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;
    
    oneapi::mkl::blas::column_major::gemm(
        queue, transA, transB, m_dim, n_dim, k, 
        alpha.to<double>(), A, lda, B, ldb, 
        beta.to<double>(), C, ldc);
  }
  
  copy_result_to_output_f64(out, result);
  
  return out;
}

// Forward declaration for optimized addmv implementation
Tensor& addmv_f64_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

//#endif // USE_ONEMKL_XPU

// result = beta * self + alpha * (mat1 * mat2)
Tensor& addmm_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  checkBackend("addmm_out", {result, self, mat1, mat2}, Backend::XPU);
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");
  TORCH_CHECK(
      mat1.dtype() == mat2.dtype(),
      "expected mat1 and mat2 to have the same dtype, but got: ",
      mat1.dtype(),
      " != ",
      mat2.dtype())

  // complex case
  if (self.is_complex()) {
    at::native::addmm_complex_out_xpu(self, mat1, mat2, beta, alpha, result);

    return result;
  }

  std::vector<int64_t> result_shape = {mat1.size(0), mat2.size(1)};
  result.resize_(result_shape);

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  if (mat1.numel() == 0) {
    if (beta.to<float>() == 0.f) {
      return result.zero_();
    }
    return at::mul_out(
        result,
        self.expand(result.sizes()),
        at::native::scalar_tensor(
            beta, self.scalar_type(), std::nullopt, at::kCPU, std::nullopt));
  }

  TORCH_CHECK(
      are_expandable(self.sizes(), result_shape),
      "addmm_out input must be expanable to:",
      result_shape,
      " but got:",
      self.sizes());

  // Different float64 handling.

  print_logs();
  count_addmm_out++;
  
  // Debug: Print stack info to understand call patterns
  if (count_addmm_out == 18176) {
    std::cout << "ADDMM call #" << count_addmm_out << " - dtype: " 
              << (mat1.scalar_type() == at::kDouble ? "float64" : "other") 
              << ", sizes: [" << mat1.size(0) << "x" << mat1.size(1) << "] @ [" 
              << mat2.size(0) << "x" << mat2.size(1) << "]" << std::endl;
  }

//#if defined(USE_ONEMKL_XPU)
  if (mat1.scalar_type() == at::kDouble) {
    // Use oneMKL GEMM for float64 which supports alpha/beta directly
    return at::native::xpu::addmm_f64_out_xpu_mkl(self, mat1, mat2, beta, alpha, result);
  }
//#endif // USE_ONEMKL_XPU
//     // Fallback to current oneDNN approach
//     bool is_inplace = self.is_same(result);
//     bool beta_not_zero = beta.to<double>() != 0.0;
//     Tensor self_copy;

//     if (is_inplace && beta_not_zero) {
//       self_copy = self.clone();
//     }

//     onednn::matmul(result, mat1, mat2, Tensor(), true, onednn::Attr());

//     if (alpha.to<double>() != 1.0) {
//       result.mul_(alpha);
//     }

//     if (beta_not_zero) {
//       if (is_inplace) {
//         result.add_(self_copy, beta);
//       } else {
//         result.add_(self, beta);
//       }
//     }

//     return result;
// #endif // USE_ONEMKL_XPU
//   }

  // general case
  Tensor bias = Tensor();
  onednn::Attr attr;
  float beta_ = beta.to<float>();
  float alpha_ = beta_ == 0.f ? alpha.to<float>() : alpha.to<float>() / beta_;
  if (beta_ == 0.f) {
    attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
  } else if (alpha_ == 1.f && beta_ == 1.f && !result.is_same(self)) {
    // if result and self are the same tensor, we use post op sum.
    bias = self;
  } else {
    Tensor binary = self.dim() < 1 ? self.unsqueeze(0) : self;
    binary = binary.dim() == 1 ? binary.unsqueeze(0) : binary;
    bool inplace = binary.is_same(result);
    if (inplace) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
      attr.append_post_sum(beta_);
    } else {
      if (at::native::onednn::is_broadcast(binary)) {
        at::native::onednn::undo_broadcast(binary);
      }
      // in test_addmv_rowmajor_colmajor_incx_incy_lda, binary is a tensor with
      // shape (5, 1) but stride(2, 2)
      binary = at::native::onednn::is_onednn_matmul_strides(binary)
          ? binary
          : binary.contiguous();
      // Tensor binary = self.expand_as(result);
      // For post-binary-add, onednn needs binary scale=1.f
      // Thus we need the following transformation
      // alpha * matmul(mat1, mat2) + beta * binary
      // beta * (alpha/beta * matmul(src, wei) + binary)
      attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
      attr.append_post_binary<true>(attr.kind_with_binary_add, binary);
      attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
    }
  }
  onednn::matmul(result, mat1, mat2, bias, true, attr);
  return result;
}

Tensor& _addmm_activation_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    bool use_gelu,
    at::Tensor& result) {
  addmm_out(self, mat1, mat2, beta, alpha, result);
  if (use_gelu) {
    at::gelu_(result);
  } else {
    at::relu_(result);
  }
  return result;
}

Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& result) {
  checkBackend("mm_out", {result, self, mat2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0],
      "x",
      self.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");
  TORCH_CHECK(
      self.dtype() == mat2.dtype(),
      "expected mat1 and mat2 to have the same dtype, but got: ",
      self.dtype(),
      " != ",
      mat2.dtype())

  result.resize_({self.size(0), mat2.size(1)});
  if (self.numel() == 0 || mat2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  if (self.is_complex()) {
    at::native::mm_complex_out_xpu(self, mat2, result);

    return result;
  }

  onednn::matmul(result, self, mat2, Tensor(), true, onednn::Attr());
  return result;
}

// result = beta * input + alpha * (batch1 @ batch2)
Tensor& baddbmm_out(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  checkBackend("baddbmm_out", {input, batch1, batch2}, Backend::XPU);
  TORCH_CHECK(batch1.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  std::vector<int64_t> result_shape = {
      batch1.size(0), batch1.size(1), batch2.size(2)};
  result.resize_(result_shape);
  if (result.numel() == 0) {
    return result;
  } else if (batch1.size(2) == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      at::mul_out(result, input, beta);
      return result;
    }
  }

  TORCH_CHECK(
      are_expandable(input.sizes(), result_shape),
      "baddbmm_out input must be expanable to:",
      result_shape,
      " but got:",
      input.sizes());

  // complex case
  if (input.is_complex()) {
    at::native::baddbmm_complex_out_xpu(
        input, batch1, batch2, beta, alpha, result);

    return result;
  }

  print_logs();
  count_baddbmm_out++;

  // Different float64 handling.
  if (batch1.scalar_type() == at::kDouble || batch2.scalar_type() == at::kDouble) {
    // Use oneMKL BLAS for float64 which supports alpha/beta directly
    return at::native::xpu::baddbmm_f64_out_xpu_mkl(input, batch1, batch2, beta, alpha, result);
  }

  // general case
  onednn::Attr attr;
  float beta_ = beta.to<float>();
  float alpha_ = beta_ == 0.f ? alpha.to<float>() : alpha.to<float>() / beta_;
  Tensor binary;
  if (beta_ == 0.f) {
    attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
  } else {
    Tensor binary = input.dim() < 1 ? input.unsqueeze(0) : input;
    binary = binary.dim() < 3 ? binary.unsqueeze(0) : binary;
    // If input is a 1d tensor need be broadcasted, we need unsqueeze twice.
    binary = binary.dim() < 3 ? binary.unsqueeze_(0) : binary;
    bool inplace = binary.is_same(result);
    if (inplace) {
      attr.append_post_eltwise(
          1.f, alpha.to<float>(), 0.f, attr.kind_with_linear);
      attr.append_post_sum(beta_);
    } else {
      if (at::native::onednn::is_broadcast(binary)) {
        at::native::onednn::undo_broadcast(binary);
      }
      binary = at::native::onednn::is_onednn_matmul_strides(binary)
          ? binary
          : binary.contiguous();
      attr.append_post_eltwise(1.f, alpha_, 0.f, attr.kind_with_linear);
      attr.append_post_binary<true>(attr.kind_with_binary_add, binary);
      attr.append_post_eltwise(1.f, beta_, 0.f, attr.kind_with_linear);
    }
  }
  onednn::matmul(result, batch1, batch2, at::Tensor(), true, attr);
  return result;
}

Tensor& bmm_out(const Tensor& self, const Tensor& batch2, Tensor& result) {
  checkBackend("bmm_out", {result, self, batch2}, Backend::XPU);
  TORCH_CHECK(self.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "expected 3D tensor");

  result.resize_({self.size(0), self.size(1), batch2.size(2)});
  if (self.numel() == 0 || batch2.numel() == 0) {
    if (result.numel() > 0)
      result.zero_();
    return result;
  }

  // complex case
  if (self.is_complex()) {
    at::native::bmm_complex_out_xpu(self, batch2, result);

    return result;
  }

  onednn::matmul(result, self, batch2, at::Tensor(), true, onednn::Attr());
  return result;
}

Tensor bmm(const Tensor& self, const Tensor& batch2) {
  auto result = at::empty({0}, self.options());
  at::native::xpu::bmm_out(self, batch2, result);
  return result;
}

Tensor& addmv_out(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  count_addmv_out++;
  
  TORCH_CHECK(
      (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
      "vector + matrix @ vector expected, got ",
      self.dim(),
      ", ",
      mat.dim(),
      ", ",
      vec.dim());

  // Size validation
  if (self.dim() == 1 && self.size(0) != 1) {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
        "size mismatch, get ",
        self.size(0),
        ", ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
  } else {
    TORCH_CHECK(
        (mat.size(1) == vec.size(0)),
        "size mismatch, get ",
        mat.size(0),
        "x",
        mat.size(1),
        ",",
        vec.size(0));
  }
  
  // For float64, use optimized oneMKL gemv implementation
  if (out.scalar_type() == at::kDouble || 
      mat.scalar_type() == at::kDouble || 
      vec.scalar_type() == at::kDouble) {
    return addmv_f64_out_xpu_mkl(self, mat, vec, beta, alpha, out);
  }
  
  // For other types, fall back to addmm-based implementation
  Tensor self_v;
  if (self.dim() == 1 && self.size(0) != 1) {
    self_v = self.view({self.size(0), 1});
  } else {
    self_v = self;
  }

  Tensor vec_v = vec.view({vec.size(0), 1});

  // Use addmm_out which now handles float64 with proper oneMKL alpha/beta scaling
  at::native::xpu::addmm_out(self_v, mat, vec_v, beta, alpha, out);

  out.resize_({mat.size(0)});
  return out;
}

// Optimized oneMKL implementation for float64 addmv operation using gemv
Tensor& addmv_f64_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  
  const int64_t m = mat.size(0);
  const int64_t n = mat.size(1);
  
  // Prepare result vector - ensure it's the right size and type
  Tensor result;
  if (self.dim() == 0 || (self.dim() == 1 && self.size(0) == 1)) {
    // Scalar or single element - create appropriately sized result vector
    result = at::zeros({m}, at::device(self.device()).dtype(at::kDouble));
    if (self.dim() == 1 && self.size(0) == 1) {
      result.fill_(self.item().to<double>());
    } else if (self.dim() == 0) {
      result.fill_(self.item().to<double>());
    }
  } else {
    // Vector input - ensure proper type and contiguity
    result = self.scalar_type() == at::kDouble ? 
             self.contiguous().clone().detach() : 
             self.to(at::kDouble).contiguous().clone().detach();
  }
  
  // Process input matrix - ensure float64 and contiguous
  Tensor a = mat.scalar_type() == at::kDouble ? 
             mat.contiguous() : 
             mat.to(at::kDouble).contiguous();
  
  // Process input vector - ensure float64 and contiguous
  Tensor x = vec.scalar_type() == at::kDouble ? 
             vec.contiguous() : 
             vec.to(at::kDouble).contiguous();
  
  // Set up oneMKL gemv parameters
  auto queue = c10::xpu::getCurrentXPUStream().queue();
  
  // For column-major gemv: A is m x n, lda should be >= m
  // We assume PyTorch tensors are row-major, so we need to transpose
  const oneapi::mkl::transpose trans = oneapi::mkl::transpose::T;
  const int64_t gemv_m = n;  // After transpose, rows become cols
  const int64_t gemv_n = m;  // After transpose, cols become rows
  const int64_t lda = std::max(int64_t{1}, n);  // Leading dim of original matrix
  const int64_t incx = x.stride(0);
  const int64_t incy = result.stride(0);
  
  const double alpha_val = alpha.to<double>();
  const double beta_val = beta.to<double>();
  
  const double* A = a.data_ptr<double>();
  const double* X = x.data_ptr<double>();
  double* Y = result.data_ptr<double>();
  
  // Call oneMKL gemv: y = alpha * A^T * x + beta * y
  // Where A^T transforms row-major to column-major semantics
  oneapi::mkl::blas::column_major::gemv(
      queue, trans, gemv_m, gemv_n, 
      alpha_val, A, lda, 
      X, incx, 
      beta_val, Y, incy);
  
  // Copy result back to output tensor
  if (out.scalar_type() == at::kDouble) {
    out.copy_(result);
  } else {
    out.copy_(result.to(out.scalar_type()));
  }
  
  return out;
}

Tensor& tensordot_out(
    const Tensor& input1,
    const Tensor& input2,
    IntArrayRef dims1,
    IntArrayRef dims2,
    Tensor& result) {
  Tensor result_tmp = at::tensordot(input1, input2, dims1, dims2);
  auto result_dtype = result_tmp.scalar_type();
  auto output_tensor_dtype = result.scalar_type();
  auto output_device = result.device();
  auto input1_device = input1.device();
  auto input2_device = input2.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
      (output_device == input1_device) && (input1_device == input2_device),
      "tensordot: Expected the output and input tensors to be on the "
      "same device, but got the output tensor on ",
      output_device,
      ", input tensor a on ",
      input1_device,
      ", and input tensor b on ",
      input2_device);
  // check if the computed result has the same dtype as the out tensor
  // (because tensordot does not support type promotion)
  TORCH_CHECK(
      result_dtype == output_tensor_dtype,
      "tensordot",
      ": Expected the output tensor to have dtype ",
      result_dtype,
      ", but got an output tensor with dtype ",
      output_tensor_dtype);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("tensordot.out", TORCH_FN(tensordot_out));
}
} // namespace xpu

TORCH_IMPL_FUNC(addmm_out_xpu)
(const Tensor& self,
 const Tensor& mat1,
 const Tensor& mat2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::addmm_out(self, mat1, mat2, beta, alpha, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(mm_out_xpu)
(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::mm_out(self, mat2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(bmm_out_xpu)
(const Tensor& self, const Tensor& batch2, const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::bmm_out(self, batch2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(addmm_activation_out_xpu)
(const Tensor& self,
 const Tensor& mat1,
 const Tensor& mat2,
 const Scalar& beta,
 const Scalar& alpha,
 bool use_gelu,
 const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::_addmm_activation_out(
      self, mat1, mat2, beta, alpha, use_gelu, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(baddbmm_out_xpu)
(const Tensor& self,
 const Tensor& batch1,
 const Tensor& batch2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  xpu::baddbmm_out(
      self,
      batch1,
      batch2,
      beta,
      alpha,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(addmv_out_xpu)
(const Tensor& self,
 const Tensor& mat,
 const Tensor& vec,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  xpu::addmv_out(self, mat, vec, beta, alpha, const_cast<Tensor&>(result));
}

Tensor _weight_int4pack_mm_xpu(
    const Tensor& A,
    const Tensor& B,
    int64_t qGroupSize,
    const Tensor& qScale,
    const Tensor& qZeros) {
  auto M = A.size(0); // M
  auto N = B.size(0); // N1=LCM(N, K)
  TORCH_CHECK(
      A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
      __func__,
      " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == kInt, __func__, " : expect B to be int32 tensor.");
  TORCH_CHECK(
      qZeros.dtype() == kChar,
      __func__,
      " : expect qZeros to be int8 tensor currently.");
  TORCH_CHECK(B.dim() == 2, __func__, " : expect B to 2d tensor.");

  TORCH_CHECK(
      qGroupSize > 1 && qGroupSize % 32 == 0,
      __func__,
      " : expect qGroupSize to be multiple of 32 and greater than 1, got ",
      qGroupSize);

  TORCH_CHECK(
      qScale.dim() == 2 && qScale.size(1) == N,
      __func__,
      ": expect qScale to be 2d tensor with sizes [:, ",
      N,
      "]");
  TORCH_CHECK(
      qZeros.dim() == 2 && qZeros.size(1) == N,
      __func__,
      ": expect qZeros to be 2d tensor with sizes [:, ",
      N,
      "]");

  auto C = at::empty({M, N}, A.options());

  // qscale:[K/qGroupSize, N]
  // qzp:[K/qGroupSize, N]
  at::native::onednn::woq_matmul_int4(C, A, B, qScale, qZeros, qGroupSize);

  return C;
}

Tensor& _int_mm_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& result) {
  TORCH_CHECK(
      self.dim() == 2,
      "Expected self to be of dimension 2 but got ",
      self.dim());
  TORCH_CHECK(
      mat2.dim() == 2,
      "Expected mat2 to be of dimension 2 but got ",
      mat2.dim());
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "self.size(1) needs to match mat2.size(0) but got ",
      self.size(1),
      " and ",
      mat2.size(0));

  TORCH_CHECK(
      self.dtype() == at::kChar,
      "Expected self dtype to be of type int8 but got ",
      self.dtype());
  TORCH_CHECK(
      mat2.dtype() == at::kChar,
      "Expected mat2 dtype to be of type int8 but got ",
      mat2.dtype());
  TORCH_CHECK(
      result.dtype() == at::kInt,
      "Expected result dtype to be of type kInt but got ",
      result.dtype());
  TORCH_CHECK(
      result.size(0) == self.size(0),
      "Expected result.size(0) to be ",
      self.size(0),
      " but got ",
      result.size(0));
  TORCH_CHECK(
      result.size(1) == mat2.size(1),
      "Expected result.size(1) to be ",
      mat2.size(1),
      " but got ",
      result.size(1));

  TORCH_CHECK(
      result.dim() == 2,
      "Expected result to be of dimension 2 but got ",
      result.dim());

  TORCH_CHECK(result.is_contiguous(), "Expected result to be contiguous.");

  if (result.numel() == 0 || self.size(1) == 0) {
    return result.zero_();
  }

  Tensor bias = at::Tensor();
  Tensor mat2_scales = at::ones({1}, mat2.options().dtype(at::kFloat));
  Tensor mat2_zero_points = at::Tensor();
  auto post_op_args = torch::List<std::optional<at::Scalar>>();

  at::native::onednn::quantized_matmul(
      self.contiguous(),
      1.0,
      0,
      mat2,
      mat2_scales,
      mat2_zero_points,
      bias,
      result,
      1.0,
      0,
      result.scalar_type(),
      /*other*/ std::nullopt,
      /*other scale*/ 1.0,
      /*other zp*/ 0,
      /*binary post op*/ "none",
      /*binary alpha*/ 1.0,
      /*post_op_name*/ "none",
      post_op_args,
      /*post_op_algorithm*/ "none",
      /*m2_trans*/ true);
  return result;
}

Tensor _int_mm_xpu(const Tensor& self, const Tensor& mat2) {
  Tensor result =
      at::empty({self.size(0), mat2.size(1)}, self.options().dtype(at::kInt));
  return _int_mm_out_xpu(self, mat2, result);
}

Tensor _weight_int8pack_mm_xpu(
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(
      A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
      " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");
  TORCH_CHECK(
      A.stride(1) == 1, " : A must be contiguous on the last dimension.");
  TORCH_CHECK(B.dtype() == kChar, " : expect B to be int8 tensor.");
  TORCH_CHECK(B.is_contiguous(), " : expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == K, " : expect B.size(1) == ", K);

  TORCH_CHECK(
      scales.dim() == 1 && scales.size(0) == N,
      " : expect scales to be 1d tensor with size ",
      N);

  auto C = at::empty({M, N}, A.options());

  // --- Launch kernel ---
  Tensor bias = at::Tensor();
  Tensor mat2_zero_points = at::Tensor();
  Tensor non_const_scales = scales;
  auto post_op_args = torch::List<std::optional<at::Scalar>>();

  at::native::onednn::quantized_matmul(
      A.contiguous(),
      1.0,
      0,
      B,
      non_const_scales,
      mat2_zero_points,
      bias,
      C,
      1.0,
      0,
      C.scalar_type(),
      /*other*/ std::nullopt,
      /*other scale*/ 1.0,
      /*other zp*/ 0,
      /*binary post op*/ "none",
      /*binary alpha*/ 1.0,
      /*post_op_name*/ "none",
      post_op_args,
      /*post_op_algorithm*/ "none",
      /*m2_trans*/ false);

  return C;
}

Tensor _bmm_dtype_xpu(
    const Tensor& batch1,
    const Tensor& batch2,
    const at::ScalarType out_dtype) {
  Tensor out = at::empty(
      {batch1.size(0), batch1.size(1), batch2.size(2)},
      batch1.options().dtype(out_dtype));
  return _bmm_out_dtype_xpu(batch1, batch2, out_dtype, out);
}

static void out_dtype_checks(
    const Tensor& mat1,
    const at::ScalarType out_dtype,
    const Tensor& out) {
  TORCH_CHECK(
      out_dtype == out.scalar_type(),
      "out_dtype must be the same as the dtype of the provided out tensor");
  TORCH_CHECK(
      out_dtype == mat1.scalar_type() ||
          (out_dtype == at::ScalarType::Float &&
           (mat1.scalar_type() == at::ScalarType::Half ||
            mat1.scalar_type() == at::ScalarType::BFloat16)),
      "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");
}

Tensor& _bmm_out_dtype_xpu(
    const Tensor& batch1,
    const Tensor& batch2,
    const at::ScalarType out_dtype,
    Tensor& out) {
  out_dtype_checks(batch1, out_dtype, out);
  xpu::bmm_out(batch1, batch2, const_cast<Tensor&>(out));
  return out;
}

Tensor _baddbmm_dtype_xpu(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const at::ScalarType out_dtype,
    const Scalar& beta,
    const Scalar& alpha) {
  TORCH_CHECK(
      self.scalar_type() == out_dtype || self.scalar_type() == batch1.dtype(),
      "self dtype must match either out_dtype or batch1 dtype");
  Tensor out = at::empty(
      {batch1.size(0), batch1.size(1), batch2.size(2)},
      batch1.options().dtype(out_dtype));
  return _baddbmm_out_dtype_xpu(
      self, batch1, batch2, out_dtype, beta, alpha, out);
}

Tensor& _baddbmm_out_dtype_xpu(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const at::ScalarType out_dtype,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  out_dtype_checks(batch1, out_dtype, out);
  xpu::baddbmm_out(
      self,
      batch1,
      batch2,
      beta,
      alpha,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<Tensor&>(out));
  return out;
}

Tensor _mm_dtype_xpu(
    const Tensor& self,
    const Tensor& mat2,
    const at::ScalarType out_dtype) {
  Tensor result =
      at::empty({self.size(0), mat2.size(1)}, self.options().dtype(out_dtype));
  return _mm_dtype_out_xpu(self, mat2, out_dtype, result);
}

Tensor& _mm_dtype_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    const at::ScalarType out_dtype,
    Tensor& out) {
  out_dtype_checks(self, out_dtype, out);
  xpu::mm_out(self, mat2, const_cast<Tensor&>(out));
  return out;
}

Tensor _addmm_dtype_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const at::ScalarType out_dtype,
    const Scalar& beta,
    const Scalar& alpha) {
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  Tensor result =
      at::empty({mat1.size(0), mat2.size(1)}, self.options().dtype(out_dtype));
  return _addmm_dtype_out_xpu(self, mat1, mat2, out_dtype, beta, alpha, result);
}

Tensor& _addmm_dtype_out_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const at::ScalarType out_dtype,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  out_dtype_checks(mat1, out_dtype, out);
  xpu::addmm_out(self, mat1, mat2, beta, alpha, out);
  return out;
}

} // namespace at::native
