#include "gemm/backend.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

namespace gemm {
using Row=std::vector<std::pair<std::string,std::string>>;
template<class T> std::string str(T v) { std::ostringstream s; s<<std::setprecision(12)<<v; return s.str(); }
std::string env(const char* name) { const char* v=std::getenv(name); return v?v:""; }
std::string quote(const std::string& s) {
  std::string out="\""; for(char ch:s) { if(ch=='\"') out+='\"'; out+=ch; } return out+'\"';
}
std::string csv_line(const Row& row,bool header) {
  std::string out;
  for(const auto& field:row) { if(!out.empty()) out+=','; out+=quote(header?field.first:field.second); }
  return out;
}
void append_csv(const std::string& path,const Row& row) {
  if(path.empty()) return;
  std::filesystem::path p(path);
  if(!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path());
  bool exists=std::filesystem::exists(p) && std::filesystem::file_size(p)>0;
  if(exists) {
    std::ifstream f(p); std::string header; std::getline(f,header);
    if(!header.empty() && header.back()=='\r') header.pop_back();
    if(header!=csv_line(row,true)) throw std::runtime_error("CSV schema mismatch; use a new file: "+path);
  }
  std::ofstream f(p,std::ios::app);
  if(!f) throw std::runtime_error("Cannot open CSV: "+path);
  if(!exists) f<<csv_line(row,true)<<'\n';
  f<<csv_line(row,false)<<'\n'; f.flush();
  if(!f) throw std::runtime_error("Cannot write CSV: "+path);
}
struct Device {
  cudaDeviceProp prop{};
  std::string name="UNAVAILABLE",uuid="UNAVAILABLE",reason;
  int runtime=0,driver=0,blas=0;
};
Device detect(int index) {
  Device d;
  CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime));
  cudaError_t err=cudaDriverGetVersion(&d.driver);
  if(err!=cudaSuccess) { d.reason=cudaGetErrorString(err); return d; }
  int count=0; err=cudaGetDeviceCount(&count);
  if(err!=cudaSuccess || count==0) {
    d.reason=err==cudaSuccess ? "No CUDA GPU detected" : cudaGetErrorString(err);
    (void)cudaGetLastError(); return d;
  }
  if(index>=count) throw std::invalid_argument("device index is outside detected device count");
  CUDA_CHECK(cudaSetDevice(index)); CUDA_CHECK(cudaGetDeviceProperties(&d.prop,index));
  d.name=d.prop.name;
  std::ostringstream uuid; uuid<<std::hex<<std::setfill('0');
  for(unsigned char b:d.prop.uuid.bytes) uuid<<std::setw(2)<<static_cast<unsigned>(b);
  d.uuid=uuid.str();
  BlasHandle h; BLAS_CHECK(cublasGetVersion(h.value,&d.blas));
  return d;
}
struct Result {
  std::string status="SKIP",reason;
  std::vector<double> times;
  double mean=0,median=0,min=0,max=0,p95=0,stddev=0,tflops=0;
  double max_abs=0,rel_l2=0,atol=0,rtol=0;
  std::size_t checked=0,bad=0;
  AlgorithmInfo algo;
};
std::size_t mul(std::size_t a,std::size_t b) {
  if(b && a>std::numeric_limits<std::size_t>::max()/b) throw Skip("Matrix byte count overflow");
  return a*b;
}
std::size_t add(std::size_t a,std::size_t b) {
  if(a>std::numeric_limits<std::size_t>::max()-b) throw Skip("Memory requirement overflow");
  return a+b;
}
// Counter-based deterministic inputs: exact float fractions before type conversion.
// Independent salts avoid accidental A/B/C correlation. No framework RNG needed.
std::uint64_t mix(std::uint64_t x) {
  x+=0x9e3779b97f4a7c15ull; x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;
  x=(x^(x>>27))*0x94d049bb133111ebull; return x^(x>>31);
}
std::vector<float> data(std::size_t n,std::uint64_t seed) {
  std::vector<float> v(n);
  for(std::size_t i=0;i<n;++i) v[i]=float(mix(i^seed)>>40)*(1.0f/16777216.0f)-0.5f;
  return v;
}
void validate(const Config& c,const std::vector<float>& a,const std::vector<float>& b,
              const std::vector<float>& c0,const std::vector<float>& got,Result& r) {
  const std::string out=output_name(c);
  r.atol=c.atol>=0 ? c.atol : out=="BF16" ? 0.004 : out=="FP16" ? 0.0005 : 1e-5;
  r.rtol=c.rtol>=0 ? c.rtol : out=="BF16" ? 0.008 : out=="FP16" ? 0.001 : 2e-5;
  const std::size_t total=got.size();
  const bool full=c.check_all || total<=4096;
  const std::size_t count=full ? total : std::min(total,c.check_samples);
  std::set<std::size_t> sampled;
  if(!full) {
    for(std::size_t i:{std::size_t(0),std::size_t(c.n-1),total-c.n,total-1,total/2})
      if(sampled.size()<count) sampled.insert(i);
    for(std::uint64_t i=0;sampled.size()<count;++i) sampled.insert(mix(i^c.seed)%total);
  }
  // Detect NaN/Inf anywhere, even outside the sampled reference positions.
  for(float x:got) if(!std::isfinite(x)) { r.reason="Non-finite output"; r.bad=1; r.status="FAIL"; return; }
  double err_sq=0,ref_sq=0;
  auto check=[&](std::size_t idx) {
    const bool column_major=c.layout=="column_major";
    std::size_t row=column_major?idx%c.m:idx/c.n;
    std::size_t col=column_major?idx/c.m:idx%c.n;
    double sum=0,sum_abs=0;
    for(int k=0;k<c.k;++k) {
      const std::size_t ai=column_major
          ? (c.ta?std::size_t(k)+row*c.lda():row+std::size_t(k)*c.lda())
          : (c.ta?std::size_t(k)*c.lda()+row:row*c.lda()+k);
      const std::size_t bi=column_major
          ? (c.tb?col+std::size_t(k)*c.ldb():std::size_t(k)+col*c.ldb())
          : (c.tb?col*c.ldb()+k:std::size_t(k)*c.ldb()+col);
      const double av=a[ai];
      const double bv=b[bi];
      const double product=av*bv; sum+=product; sum_abs+=std::abs(product);
    }
    const double ref=double(c.alpha)*sum+double(c.beta)*c0[idx];
    const double error=std::abs(double(got[idx])-ref);
    // Cancellation-aware error budget for FP32 accumulation. TF32 also allows
    // the bounded rounding of both FP32 operands to a 10-bit mantissa.
    const double accum=8*std::numeric_limits<float>::epsilon();
    const double input_round=c.dtype==Dtype::TF32 ? 0.000977 : 0.0;
    const double bound=r.atol+r.rtol*std::abs(ref)+(accum+input_round)*std::abs(c.alpha)*sum_abs;
    if(!std::isfinite(ref) || error>bound) ++r.bad;
    r.max_abs=std::max(r.max_abs,error); err_sq+=error*error; ref_sq+=ref*ref; ++r.checked;
  };
  if(full) for(std::size_t idx=0;idx<total;++idx) check(idx);
  else for(auto idx:sampled) check(idx);
  r.rel_l2=std::sqrt(err_sq/std::max(ref_sq,1e-30));
  r.status=r.bad ? "FAIL" : "PASS";
  if(r.bad) r.reason="FP64 reference mismatch in "+str(r.bad)+" checked elements";
}
void statistics(const Config& c,Result& r) {
  auto sorted=r.times; std::sort(sorted.begin(),sorted.end());
  r.mean=std::accumulate(sorted.begin(),sorted.end(),0.0)/sorted.size();
  std::size_t n=sorted.size(); r.median=n%2 ? sorted[n/2] : (sorted[n/2-1]+sorted[n/2])*0.5;
  r.min=sorted.front();r.max=sorted.back();r.p95=sorted[static_cast<std::size_t>(std::ceil(n*0.95))-1];
  for(double t:sorted) r.stddev+=(t-r.mean)*(t-r.mean);
  r.stddev=std::sqrt(r.stddev/n);
  r.tflops=(2.0*c.m*c.n*c.k)/(r.mean*1e9);
}
void run_case(const Config& c,const Device& d,Result& r) {
  if(!d.reason.empty()) throw Skip(d.reason);
  if(c.dtype==Dtype::FP16 && d.prop.major<7) throw Skip("FP16 Tensor Core baseline requires SM70+");
  if((c.dtype==Dtype::BF16 || c.dtype==Dtype::TF32) && d.prop.major<8) throw Skip("BF16/TF32 requires SM80+");
  if(c.dtype==Dtype::TF32 && env("NVIDIA_TF32_OVERRIDE")=="0") throw Skip("NVIDIA_TF32_OVERRIDE=0 disables requested TF32");
  std::size_t na=mul(c.m,c.k),nb=mul(c.k,c.n),nc=mul(c.m,c.n);
  std::size_t ba=mul(na,type_size(input_type(c))),bb=mul(nb,type_size(input_type(c))),bc=mul(nc,type_size(output_type(c)));
  std::size_t required=add(add(ba,bb),mul(bc,c.beta==0?1:2));
  if(c.backend=="cublaslt") required=add(required,c.workspace);
  std::size_t free=0,total=0; CUDA_CHECK(cudaMemGetInfo(&free,&total));
  // Leave room for lazy library modules and context allocations.
  if(required>free || free-required<std::min<std::size_t>(256ull<<20,total/10))
    throw Skip("Insufficient device memory: need "+str(required)+" bytes plus reserve; free="+str(free));
  Stream s;
  Buffer a(ba),b(bb),out(bc),initial(c.beta==0?0:bc);
  auto ha=data(na,c.seed^0xa123ull),hb=data(nb,c.seed^0xb456ull),hc=data(nc,c.seed^0xc789ull);
  upload(a.value,ha,input_type(c),s.value); upload(b.value,hb,input_type(c),s.value);
  upload(out.value,hc,output_type(c),s.value);
  if(c.beta!=0) CUDA_CHECK(cudaMemcpyAsync(initial.value,out.value,bc,cudaMemcpyDeviceToDevice,s.value));
  std::unique_ptr<GemmPlan> plan;
  if(c.backend=="cublas") plan=make_cublas(c,a.value,b.value,out.value,s.value);
  else plan=make_cublaslt(c,a.value,b.value,out.value,initial.value,bc,s.value);
  r.algo=plan->info;
  auto reset=[&] { if(c.beta!=0) CUDA_CHECK(cudaMemcpyAsync(out.value,initial.value,bc,cudaMemcpyDeviceToDevice,s.value)); };
  // Mandatory untimed call resolves lazy initialization even with warmup=0.
  reset();plan->run();CUDA_CHECK(cudaStreamSynchronize(s.value));
  Graph graph;
  if(c.mode=="graph") {
    CUDA_CHECK(cudaStreamBeginCapture(s.value,cudaStreamCaptureModeThreadLocal));
    try { for(int j=0;j<c.effective_inner();++j) plan->run(); }
    catch(...) { cudaGraph_t discarded=nullptr; cudaStreamEndCapture(s.value,&discarded); if(discarded) cudaGraphDestroy(discarded); throw; }
    CUDA_CHECK(cudaStreamEndCapture(s.value,&graph.graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
  }
  auto batch=[&] {
    if(c.mode=="graph") CUDA_CHECK(cudaGraphLaunch(graph.executable,s.value));
    else for(int j=0;j<c.effective_inner();++j) plan->run();
  };
  // Always upload / prime a graph outside timing.
  if(c.mode=="graph") { reset();batch();CUDA_CHECK(cudaStreamSynchronize(s.value)); }
  for(int w=0;w<c.warmup;++w) { reset();batch(); }
  CUDA_CHECK(cudaStreamSynchronize(s.value));
  Event start,stop; r.times.reserve(c.repeats);
  for(int repeat=0;repeat<c.repeats;++repeat) {
    reset(); // Stream ordering puts the reset BEFORE the start event.
    CUDA_CHECK(cudaEventRecord(start.value,s.value));
    batch();
    CUDA_CHECK(cudaEventRecord(stop.value,s.value));
    CUDA_CHECK(cudaEventSynchronize(stop.value)); // Once per sample, never per GEMM.
    float elapsed=0; CUDA_CHECK(cudaEventElapsedTime(&elapsed,start.value,stop.value));
    if(!(elapsed>0) || !std::isfinite(elapsed)) throw std::runtime_error("Invalid CUDA event duration");
    r.times.push_back(double(elapsed)/c.effective_inner());
  }
  statistics(c,r);
  auto got=download(out.value,nc,output_type(c),s.value);
  // beta=0: every output is identical. beta!=0: effective_inner=1 and C restored.
  validate(c,ha,hb,hc,got,r);
}
Row row_for(const Config& c,const Device& d,const Result& r) {
  auto time=[&](double x) { return r.times.empty()?std::string():str(x); };
  std::string samples; for(double t:r.times) { if(!samples.empty()) samples+=';'; samples+=str(t); }
  return {
    {"Schema","gemm-v1"},{"GPU",d.name},{"GPU_UUID",d.uuid},{"SM",str(d.prop.major*10+d.prop.minor)},
    {"SM_Count",str(d.prop.multiProcessorCount)},{"VRAM_bytes",str(d.prop.totalGlobalMem)},
    {"CUDA_Compile",str(CUDART_VERSION)},{"CUDA_Runtime",str(d.runtime)},{"CUDA_Driver",str(d.driver)},{"cuBLAS_Version",str(d.blas)},
    {"Backend",c.backend},{"Dtype",dtype_name(c.dtype)},{"InputDtype",c.dtype==Dtype::TF32?"FP32":dtype_name(c.dtype)},
    {"OutputDtype",output_name(c)},{"Compute",compute_name(c)},{"Reduction","FP32"},{"Layout",c.layout},
    {"M",str(c.m)},{"N",str(c.n)},{"K",str(c.k)},{"TransA",c.ta?"T":"N"},{"TransB",c.tb?"T":"N"},
    {"LDA",str(c.lda())},{"LDB",str(c.ldb())},{"LDC",str(c.ldc())},{"Alpha",str(c.alpha)},{"Beta",str(c.beta)},
    {"Mode",c.mode},{"TimingProtocol","cuda_event_batch_v1"},{"CachePolicy","reuse_operands"},
    {"BetaPolicy",c.beta==0?"overwrite":"reset_outside_timing"},{"Warmup",str(c.warmup)},
    {"Repeats",str(c.repeats)},{"InnerRequested",str(c.inner)},{"Inner",str(c.effective_inner())},{"Seed",str(c.seed)},
    {"Time_ms",time(r.mean)},{"Time_us",time(r.mean*1000)},{"Median_ms",time(r.median)},{"Min_ms",time(r.min)},
    {"Max_ms",time(r.max)},{"P95_ms",time(r.p95)},{"Stddev_ms",time(r.stddev)},{"TFLOPS",time(r.tflops)},
    {"Status",r.status},{"Reason",r.reason},{"CheckMode",c.check_all || std::size_t(c.m)*c.n<=4096?"full":"sampled"},
    {"Checked",str(r.checked)},{"Bad",str(r.bad)},{"MaxAbsError",r.checked?str(r.max_abs):""},
    {"RelativeL2",r.checked?str(r.rel_l2):""},{"Atol",r.checked?str(r.atol):""},{"Rtol",r.checked?str(r.rtol):""},
    {"WorkspaceBudget_bytes",c.backend=="cublaslt"?str(c.workspace):"0"},
    {"WorkspaceUsed_bytes",c.backend=="cublaslt"?str(r.algo.workspace_used):"library_managed"},
    {"Algorithm",r.algo.description},{"LtSelection",c.backend=="cublaslt"?c.lt_selection:""},
    {"LtHeuristicIndex",str(r.algo.heuristic_index)},{"LtCandidates",str(r.algo.candidates)},
    {"NVIDIA_TF32_OVERRIDE",env("NVIDIA_TF32_OVERRIDE")},{"CUBLAS_WORKSPACE_CONFIG",env("CUBLAS_WORKSPACE_CONFIG")},
    {"Samples_ms",samples}
  };
}
} // namespace gemm
int main(int argc,char** argv) {
  using namespace gemm;
  try {
    Config c=parse_args(argc,argv);
    if(c.help) { print_help(); return 0; }
    Device d=detect(c.device);
    std::cout<<"GPU: "<<d.name<<" | SM "<<d.prop.major<<d.prop.minor
             <<" | CUDA runtime "<<d.runtime<<" | cuBLAS "<<d.blas<<'\n';
    if(c.device_info) { if(!d.reason.empty()) std::cout<<d.reason<<'\n'; return d.reason.empty()?0:3; }
    int passed=0,failed=0,skipped=0;
    for(const auto& test:cases(c)) {
      Result r; bool fatal=false;
      try { run_case(test,d,r); }
      catch(const Skip& e) { r.status="SKIP";r.reason=e.what(); }
      catch(const std::bad_alloc&) { r.status="SKIP";r.reason="Insufficient host memory"; }
      catch(const CudaError& e) { r.status="FAIL";r.reason=e.what();fatal=true; }
      catch(const std::exception& e) { r.status="FAIL";r.reason=e.what(); }
      append_csv(test.csv,row_for(test,d,r));
      std::cout<<test.backend<<' '<<dtype_name(test.dtype)<<' '<<test.m<<'x'<<test.n<<'x'<<test.k<<' '
               <<(test.ta?'T':'N')<<(test.tb?'T':'N')<<' '<<test.mode<<' '<<r.status;
      if(!r.times.empty()) std::cout<<" mean="<<r.mean<<" ms median="<<r.median<<" ms min="<<r.min<<" ms "<<r.tflops<<" TFLOPS";
      if(!r.reason.empty()) std::cout<<" ("<<r.reason<<')';
      std::cout<<'\n';
      if(r.status=="PASS") ++passed; else if(r.status=="SKIP") ++skipped; else ++failed;
      if(fatal) { std::cerr<<"CUDA execution error: stopping this process; batch script isolates subsequent cases.\n"; break; }
    }
    std::cout<<"PASS="<<passed<<" FAIL="<<failed<<" SKIP="<<skipped<<'\n';
    return failed?1:passed?0:3;
  } catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n';return 2; }
}


