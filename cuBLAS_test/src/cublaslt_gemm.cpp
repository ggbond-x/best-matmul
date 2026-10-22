#include "gemm/backend.hpp"
#include <cublasLt.h>
#include <algorithm>
#include <limits>
#include <sstream>

namespace gemm {
namespace {
struct LtHandle {
  cublasLtHandle_t v=nullptr;
  LtHandle() { BLAS_CHECK(cublasLtCreate(&v)); }
  ~LtHandle() { if(v) cublasLtDestroy(v); }
};
struct Desc {
  cublasLtMatmulDesc_t v=nullptr;
  ~Desc() { if(v) cublasLtMatmulDescDestroy(v); }
};
struct Layout {
  cublasLtMatrixLayout_t v=nullptr;
  ~Layout() { if(v) cublasLtMatrixLayoutDestroy(v); }
  void create(cudaDataType_t type,int rows,int cols,int ld,cublasLtOrder_t order) {
    BLAS_CHECK(cublasLtMatrixLayoutCreate(&v,type,rows,cols,ld));
    BLAS_CHECK(cublasLtMatrixLayoutSetAttribute(v,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order)));
  }
};
struct Preference {
  cublasLtMatmulPreference_t v=nullptr;
  Preference() { BLAS_CHECK(cublasLtMatmulPreferenceCreate(&v)); }
  ~Preference() { if(v) cublasLtMatmulPreferenceDestroy(v); }
};
int attribute(const cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoConfigAttributes_t attr) {
  int value=0;std::size_t written=0;
  BLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo,attr,&value,sizeof(value),&written));
  return value;
}
std::string describe(const cublasLtMatmulAlgo_t& algo) {
  std::ostringstream s;
  s<<"id="<<attribute(algo,CUBLASLT_ALGO_CONFIG_ID)
   <<";tile="<<attribute(algo,CUBLASLT_ALGO_CONFIG_TILE_ID)
   <<";stages="<<attribute(algo,CUBLASLT_ALGO_CONFIG_STAGES_ID)
   <<";splitK="<<attribute(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
   <<";reduction="<<attribute(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
   <<";swizzle="<<attribute(algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
   <<";custom="<<attribute(algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
  std::uint64_t flags=0;std::size_t written=0;
  auto status=cublasLtMatmulAlgoCapGetAttribute(&algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&flags,sizeof(flags),&written);
  if(status==CUBLAS_STATUS_SUCCESS) s<<";numeric_flags="<<flags;
  return s.str();
}
}
class LtPlan final : public GemmPlan {
  LtHandle handle_;
  Desc desc_;
  Layout a_desc_,b_desc_,c_desc_;
  Preference pref_;
  Buffer workspace_;
  Config c_;
  const void *a_,*b_,*initial_;
  void* out_;
  std::size_t out_bytes_;
  cudaStream_t stream_;
  cublasLtMatmulAlgo_t algo_{};

  void reset() {
    if(c_.beta!=0) CUDA_CHECK(cudaMemcpyAsync(out_,initial_,out_bytes_,cudaMemcpyDeviceToDevice,stream_));
  }
  cublasStatus_t enqueue() {
    return cublasLtMatmul(handle_.v,desc_.v,&c_.alpha,a_,a_desc_.v,b_,b_desc_.v,
        &c_.beta,out_,c_desc_.v,out_,c_desc_.v,&algo_,workspace_.value,c_.workspace,stream_);
  }
  double tune() {
    // Measure the same launch/graph mode and repetition policy as the benchmark.
    Graph graph;
    const int inner=c_.effective_inner();
    if(c_.mode=="graph") {
      CUDA_CHECK(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal));
      try { for(int j=0;j<inner;++j) run(); }
      catch(...) { cudaGraph_t discard=nullptr;cudaStreamEndCapture(stream_,&discard);if(discard) cudaGraphDestroy(discard);throw; }
      CUDA_CHECK(cudaStreamEndCapture(stream_,&graph.graph));
      CUDA_CHECK(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
    }
    auto batch=[&] {
      if(c_.mode=="graph") CUDA_CHECK(cudaGraphLaunch(graph.executable,stream_));
      else for(int j=0;j<inner;++j) run();
    };
    for(int i=0;i<2;++i) { reset();batch(); }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    Event start,end;std::vector<float> ms;
    for(int i=0;i<5;++i) {
      reset();CUDA_CHECK(cudaEventRecord(start.value,stream_));batch();
      CUDA_CHECK(cudaEventRecord(end.value,stream_));CUDA_CHECK(cudaEventSynchronize(end.value));
      float t=0;CUDA_CHECK(cudaEventElapsedTime(&t,start.value,end.value));ms.push_back(t/inner);
    }
    std::sort(ms.begin(),ms.end());return ms[ms.size()/2];
  }
 public:
  LtPlan(const Config& c,const void* a,const void* b,void* out,const void* initial,
         std::size_t out_bytes,cudaStream_t stream)
    : workspace_(c.workspace),c_(c),a_(a),b_(b),initial_(initial),out_(out),out_bytes_(out_bytes),stream_(stream) {
    BLAS_CHECK(cublasLtMatmulDescCreate(&desc_.v,compute_type(c),CUDA_R_32F));
    cublasOperation_t ta=c.ta?CUBLAS_OP_T:CUBLAS_OP_N,tb=c.tb?CUBLAS_OP_T:CUBLAS_OP_N;
    BLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_.v,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    BLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_.v,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    // Descriptors describe STORED matrices, before op(). The leading
    // dimension follows the selected physical order.
    const cublasLtOrder_t order=c.layout=="column_major"?CUBLASLT_ORDER_COL:CUBLASLT_ORDER_ROW;
    a_desc_.create(input_type(c),c.ta?c.k:c.m,c.ta?c.m:c.k,c.lda(),order);
    b_desc_.create(input_type(c),c.tb?c.n:c.k,c.tb?c.k:c.n,c.ldb(),order);
    c_desc_.create(output_type(c),c.m,c.n,c.ldc(),order);
    BLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref_.v,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&c.workspace,sizeof(c.workspace)));
    // Allow no reduction or FP32 compute-type reduction, never FP16/BF16
    // output-type/in-place split-K reduction. Keep both baselines comparable.
    std::uint32_t reduction=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    BLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref_.v,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&reduction,sizeof(reduction)));
    std::vector<cublasLtMatmulHeuristicResult_t> candidates(c.lt_candidates);
    int found=0;
    BLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_.v,desc_.v,a_desc_.v,b_desc_.v,c_desc_.v,c_desc_.v,
        pref_.v,c.lt_candidates,candidates.data(),&found));
    info.candidates=found;
    if(found==0) throw Skip("cuBLASLt found no supported algorithm for this layout/type/workspace");
    if(c.lt_selection=="index" && c.lt_index>=found) throw Skip("Requested Lt heuristic index is unavailable");
    double best=std::numeric_limits<double>::infinity();
    cublasLtMatmulAlgo_t chosen{}; bool selected=false;
    for(int i=0;i<found;++i) {
      if(c.lt_selection=="index" && i!=c.lt_index) continue;
      const auto& candidate=candidates[i];
      if(candidate.state!=CUBLAS_STATUS_SUCCESS || candidate.workspaceSize>c.workspace) continue;
      cublasLtMatmulHeuristicResult_t checked{};
      auto status=cublasLtMatmulAlgoCheck(handle_.v,desc_.v,a_desc_.v,b_desc_.v,c_desc_.v,c_desc_.v,&candidate.algo,&checked);
      if(status==CUBLAS_STATUS_NOT_SUPPORTED) continue;
      BLAS_CHECK(status);
      if(checked.state!=CUBLAS_STATUS_SUCCESS || checked.workspaceSize>c.workspace) continue;
      int split=attribute(candidate.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
      int reduce=attribute(candidate.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
      if(split>1 && reduce!=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) continue;
      algo_=candidate.algo;
      reset();status=enqueue(); // Untimed probe: actual addresses can limit an algo.
      if(status==CUBLAS_STATUS_NOT_SUPPORTED) continue;
      BLAS_CHECK(status);CUDA_CHECK(cudaStreamSynchronize(stream_));
      double score=c.lt_selection=="tune"?tune():0;
      if(score<best) {
        best=score;chosen=algo_;selected=true;
        info.heuristic_index=i;info.workspace_used=checked.workspaceSize;
      }
      if(c.lt_selection!="tune") break;
    }
    if(!selected) throw Skip("No valid Lt algorithm with FP32 reduction under the requested selection");
    algo_=chosen;info.description=describe(algo_);
  }
  void run() override { BLAS_CHECK(enqueue()); }
};
std::unique_ptr<GemmPlan> make_cublaslt(const Config& c,const void* a,const void* b,void* out,
                                      const void* initial,std::size_t bytes,cudaStream_t s) {
  return std::make_unique<LtPlan>(c,a,b,out,initial,bytes,s);
}
}

