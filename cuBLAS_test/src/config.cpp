#include "gemm/config.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace gemm {
std::string dtype_name(Dtype t) {
  switch(t) { case Dtype::FP16:return "FP16"; case Dtype::BF16:return "BF16";
    case Dtype::FP32:return "FP32"; case Dtype::TF32:return "TF32"; }
  throw std::runtime_error("Invalid dtype");
}
std::string output_name(const Config& c) {
  return c.output_fp32 || c.dtype==Dtype::FP32 || c.dtype==Dtype::TF32 ? "FP32" : dtype_name(c.dtype);
}
std::string compute_name(const Config& c) {
  return c.dtype==Dtype::FP32 ? "FP32_PEDANTIC" : c.dtype==Dtype::TF32 ? "TF32_FP32_ACC" : "FP32_ACC";
}
static long long integer(const std::string& s, long long lo, long long hi) {
  std::size_t end=0; long long v=std::stoll(s,&end);
  if(end!=s.size() || v<lo || v>hi) throw std::invalid_argument("Integer out of range: "+s);
  return v;
}
static double real(const std::string& s) {
  std::size_t end=0; double v=std::stod(s,&end);
  if(end!=s.size() || !std::isfinite(v)) throw std::invalid_argument("Invalid real: "+s);
  return v;
}
Config parse_args(int argc, char** argv) {
  Config c;
  for(int i=1;i<argc;++i) {
    std::string a=argv[i];
    if(a=="--help" || a=="-h") { c.help=true; continue; }
    if(a=="--device-info") { c.device_info=true; continue; }
    if(a=="--check-all") { c.check_all=true; continue; }
    if(i+1==argc) throw std::invalid_argument("Missing value for "+a);
    std::string v=argv[++i];
    auto pos=[&] { return static_cast<int>(integer(v,1,std::numeric_limits<int>::max())); };
    if(a=="--m") c.m=pos(); else if(a=="--n") c.n=pos(); else if(a=="--k") c.k=pos();
    else if(a=="--transa" || a=="--transb") {
      if(v!="N" && v!="T" && v!="n" && v!="t") throw std::invalid_argument("Transpose must be N or T");
      (a=="--transa" ? c.ta : c.tb)=(v=="T" || v=="t");
    } else if(a=="--alpha") c.alpha=static_cast<float>(real(v));
    else if(a=="--beta") c.beta=static_cast<float>(real(v));
    else if(a=="--dtype") {
      if(v=="fp16") c.dtype=Dtype::FP16; else if(v=="bf16") c.dtype=Dtype::BF16;
      else if(v=="fp32") c.dtype=Dtype::FP32; else if(v=="tf32") c.dtype=Dtype::TF32;
      else throw std::invalid_argument("dtype: fp16|bf16|fp32|tf32");
    } else if(a=="--output-dtype") {
      if(v!="same" && v!="fp32") throw std::invalid_argument("output-dtype: same|fp32");
      c.output_fp32=(v=="fp32");
    } else if(a=="--backend") c.backend=v;
    else if(a=="--mode") c.mode=v;
    else if(a=="--layout") {
      if(v=="row_major" || v=="row-major" || v=="RowMajor") c.layout="row_major";
      else if(v=="column_major" || v=="col_major" || v=="column-major" || v=="ColMajor") c.layout="column_major";
      else throw std::invalid_argument("layout: row_major|column_major");
    }
    else if(a=="--suite") c.suite=v;
    else if(a=="--csv") c.csv=v;
    else if(a=="--device") c.device=static_cast<int>(integer(v,0,1024));
    else if(a=="--warmup") c.warmup=static_cast<int>(integer(v,0,1000000));
    else if(a=="--repeats") c.repeats=pos();
    else if(a=="--inner") c.inner=pos();
    else if(a=="--seed") c.seed=integer(v,0,std::numeric_limits<long long>::max());
    else if(a=="--check-samples") c.check_samples=integer(v,1,std::numeric_limits<int>::max());
    else if(a=="--atol") { c.atol=real(v); if(c.atol<0) throw std::invalid_argument("atol must be >=0"); }
    else if(a=="--rtol") { c.rtol=real(v); if(c.rtol<0) throw std::invalid_argument("rtol must be >=0"); }
    else if(a=="--workspace-mb") c.workspace=static_cast<std::size_t>(integer(v,0,1048576))*1024*1024;
    else if(a=="--lt-selection") c.lt_selection=v;
    else if(a=="--lt-candidates") c.lt_candidates=static_cast<int>(integer(v,1,1024));
    else if(a=="--lt-index") c.lt_index=static_cast<int>(integer(v,0,1023));
    else throw std::invalid_argument("Unknown option: "+a);
  }
  if(!std::isfinite(c.alpha) || !std::isfinite(c.beta)) throw std::invalid_argument("alpha/beta must fit FP32");
  if(c.backend!="cublas" && c.backend!="cublaslt") throw std::invalid_argument("backend: cublas|cublaslt");
  if(c.mode!="launch" && c.mode!="graph") throw std::invalid_argument("mode: launch|graph");
  if(c.lt_selection!="first" && c.lt_selection!="index" && c.lt_selection!="tune") throw std::invalid_argument("lt-selection: first|index|tune");
  if(c.lt_selection=="index" && c.lt_index>=c.lt_candidates) throw std::invalid_argument("lt-index must be < lt-candidates");
  if(!c.suite.empty() && c.suite!="all" && c.suite!="square" && c.suite!="rect" && c.suite!="llm" && c.suite!="irregular")
    throw std::invalid_argument("suite: all|square|rect|llm|irregular");
  return c;
}
std::vector<Config> cases(const Config& c) {
  if(c.suite.empty()) return {c};
  std::vector<Config> out;
  auto add=[&](int m,int n,int k) { Config d=c; d.m=m;d.n=n;d.k=k;out.push_back(d); };
  if(c.suite=="all" || c.suite=="square") for(int x:{256,512,1024,2048,4096,8192}) add(x,x,x);
  if(c.suite=="all" || c.suite=="rect") { add(1024,4096,4096); add(4096,1024,4096); add(4096,4096,1024); add(8192,4096,4096); }
  if(c.suite=="all" || c.suite=="llm") for(int m:{1,4,16,64,128,512,2048}) add(m,4096,4096);
  if(c.suite=="all" || c.suite=="irregular") { add(127,255,513); add(1000,1031,997); add(33,4097,4093); }
  return out;
}
void print_help() {
  std::cout << R"(GEMM: C = alpha * op(A) * op(B) + beta * C
  --m INT --n INT --k INT           logical M,N,K (default 1024)
  --transa N|T --transb N|T         stored matrices may be transposed
  --layout row_major|column_major   physical A/B/C layout; default row_major
  --dtype fp16|bf16|fp32|tf32       default fp16; fp32 is strict
  --output-dtype same|fp32         default same (TF32 always stores FP32)
  --backend cublas|cublaslt         default cublas
  --alpha FLOAT --beta FLOAT       default 1,0
  --warmup INT --repeats INT --inner INT   default 10,30,10
  --mode launch|graph              default launch; separately reported
  --workspace-mb INT               Lt workspace budget, default 64 MiB
  --lt-selection first|index|tune   default first; tune outside timing
  --lt-candidates INT --lt-index INT      default 32,0 (zero-based)
  --suite all|square|rect|llm|irregular    omit to run one shape
  --seed INT --device INT          default 1234,0
  --check-samples INT              default 512; includes corners
  --check-all                      CPU FP64 reference over all outputs
  --atol FLOAT --rtol FLOAT        override automatic output tolerances
  --csv PATH                      default results/benchmark.csv (append)
  --device-info                   print GPU / library versions, then exit
  --help
beta != 0 forces effective inner=1; C is restored outside every timed region.
PASS means validation passed; SKIP means unsupported/OOM; FAIL returns exit 1.
Exit 2: CLI/setup error; exit 3: every requested case was skipped.
)";
}
}

