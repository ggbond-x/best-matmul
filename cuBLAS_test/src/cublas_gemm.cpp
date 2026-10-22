#include "gemm/backend.hpp"
namespace gemm {
class CublasPlan final : public GemmPlan {
  BlasHandle handle_;
  Config c_;
  const void *a_,*b_; void* out_;
 public:
  CublasPlan(const Config& c,const void* a,const void* b,void* out,cudaStream_t stream)
      : c_(c),a_(a),b_(b),out_(out) {
    info.description="CUBLAS_GEMM_DEFAULT";
    BLAS_CHECK(cublasSetStream(handle_.value,stream));
    BLAS_CHECK(cublasSetPointerMode(handle_.value,CUBLAS_POINTER_MODE_HOST));
    // Never reduce FP32 split-K accumulators in the lower output precision.
    BLAS_CHECK(cublasSetMathMode(handle_.value,static_cast<cublasMath_t>(
        (c.dtype==Dtype::FP32 ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH) | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  }
  void run() override {
    if(c_.layout=="column_major") {
      BLAS_CHECK(cublasGemmEx(handle_.value,c_.ta?CUBLAS_OP_T:CUBLAS_OP_N,
          c_.tb?CUBLAS_OP_T:CUBLAS_OP_N,c_.m,c_.n,c_.k,&c_.alpha,
          a_,input_type(c_),c_.lda(),b_,input_type(c_),c_.ldb(),&c_.beta,
          out_,output_type(c_),c_.ldc(),compute_type(c_),CUBLAS_GEMM_DEFAULT));
    } else {
      // Row-major C is column-major C^T. C^T = op(B)^T * op(A)^T.
      // Swap operands AND M/N, retain each operand's transpose flag.
      // Stored row strides become column-major leading dimensions. No copies.
      BLAS_CHECK(cublasGemmEx(handle_.value,c_.tb?CUBLAS_OP_T:CUBLAS_OP_N,
          c_.ta?CUBLAS_OP_T:CUBLAS_OP_N,c_.n,c_.m,c_.k,&c_.alpha,
          b_,input_type(c_),c_.ldb(),a_,input_type(c_),c_.lda(),&c_.beta,
          out_,output_type(c_),c_.ldc(),compute_type(c_),CUBLAS_GEMM_DEFAULT));
    }
  }
};
std::unique_ptr<GemmPlan> make_cublas(const Config& c,const void* a,const void* b,void* out,cudaStream_t stream) {

  return std::make_unique<CublasPlan>(c,a,b,out,stream);
}
}



