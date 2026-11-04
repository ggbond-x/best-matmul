// FP16/FP32 dense GEMM runner. Linux CLOCK_MONOTONIC is shared with Python.
#include "gemm/backend.hpp"
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cublasLt.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>
using namespace gemm;
using Fields=std::map<std::string,std::string>;
template<class T> std::string text(T x) { std::ostringstream o;o<<std::setprecision(15)<<x;return o.str(); }
long long ns(clockid_t clock=CLOCK_MONOTONIC) { timespec t{};if(clock_gettime(clock,&t))throw std::runtime_error("clock_gettime");return t.tv_sec*1000000000LL+t.tv_nsec; }
std::string quoted(const std::string& s) { std::string r="\"";for(char c:s){if(c=='"')r+='"';r+=c;}return r+'"'; }
std::string json_string(const std::string& s) { std::string r="\"";for(char c:s){if(c=='"'||c=='\\')r+='\\';if(c=='\n')r+="\\n";else r+=c;}return r+'"'; }
const std::vector<std::string> raw_header={
 "run_id","case_id","phase","window","gpu_uuid","M","N","K","transA","transB","dtype","compute_type","api",
 "algorithm","algorithm_id","workspace_bytes","heuristic_candidates","tensor_core_evidence","instruction_path",
 "start_monotonic_ns","end_monotonic_ns","start_unix_ns","end_unix_ns","elapsed_ms_median","elapsed_ms_mean",
 "elapsed_ms_min","tflops","tflops_mean","inner","repeats","gpu_total_ms","wall_ms","samples_ms",
 "correctness_pass","checked_elements","max_abs_error","status","reason"};
void csv(const std::string& path,const std::vector<std::string>& head,const Fields& row) {
  bool fresh=!std::filesystem::exists(path)||!std::filesystem::file_size(path);
  std::ofstream f(path,std::ios::app);if(!f)throw std::runtime_error("Cannot open "+path);
  auto line=[&](bool header){for(std::size_t i=0;i<head.size();++i){if(i)f<<',';auto it=row.find(head[i]);f<<quoted(header?head[i]:it==row.end()?"":it->second);}f<<'\n';};
  if(fresh)line(true);line(false);f.flush();if(!f)throw std::runtime_error("Cannot write "+path);
}
struct Args {
 Config c;
 std::string id="case",run="run",phase="scan",raw,events,trials;
 double seconds=3,window=1,warmup_seconds=2;
 int minimum_samples=10;
 bool info=false, profile=false;
};
Args args(int argc,char** argv) {
 Args a;a.c.layout="column_major";a.c.lt_selection="tune";a.c.inner=20;a.c.warmup=20;a.c.check_samples=1024;
 std::set<std::string> seen;
 for(int i=1;i<argc;++i){
  std::string k=argv[i];if(!seen.insert(k).second)throw std::invalid_argument("Duplicate option: "+k);
  if(k=="--profile"){a.profile=true;continue;}
  if(k=="--info"){a.info=true;continue;}
  if(k=="--help"){std::cout<<"h800_peak_bench --info [--device N]\nOne case: --m M --n N --k K --transA N|T --transB N|T --api cublas|cublaslt --raw FILE --events FILE --trials FILE --case-id ID --run-id ID --phase scan|sustained --seconds S --window-seconds S --warmup-seconds S --warmup N --inner N --repeats N --samples N --workspace-mib N --lt-candidates N --timing launch|graph\n";std::exit(0);}
  if(++i==argc)throw std::invalid_argument("Missing value for "+k);std::string v=argv[i];
  auto integer=[&](){std::size_t p=0;int x=std::stoi(v,&p);if(p!=v.size())throw std::invalid_argument(k);return x;};
  auto real=[&](){std::size_t p=0;double x=std::stod(v,&p);if(p!=v.size()||!std::isfinite(x))throw std::invalid_argument(k);return x;};
  if(k=="--m")a.c.m=integer();else if(k=="--n")a.c.n=integer();else if(k=="--k")a.c.k=integer();
  else if(k=="--device")a.c.device=integer();
  else if(k=="--transA"||k=="--transB"){if(v!="N"&&v!="T")throw std::invalid_argument(k);(k=="--transA"?a.c.ta:a.c.tb)=v=="T";}
  else if(k=="--api")a.c.backend=v;else if(k=="--timing")a.c.mode=v;
  else if(k=="--case-id")a.id=v;else if(k=="--run-id")a.run=v;else if(k=="--phase")a.phase=v;
  else if(k=="--raw")a.raw=v;else if(k=="--events")a.events=v;else if(k=="--trials")a.trials=v;
  else if(k=="--seconds")a.seconds=real();else if(k=="--window-seconds")a.window=real();
  else if(k=="--warmup-seconds")a.warmup_seconds=real();else if(k=="--warmup")a.c.warmup=integer();
  else if(k=="--inner")a.c.inner=integer();else if(k=="--repeats")a.minimum_samples=integer();
  else if(k=="--samples"){int x=integer();if(x<=0)throw std::invalid_argument(k);a.c.check_samples=x;}
  else if(k=="--workspace-mib"){int x=integer();if(x<0||x>16384)throw std::invalid_argument(k);a.c.workspace=std::size_t(x)<<20;}
  else if(k=="--lt-candidates")a.c.lt_candidates=integer();
  else throw std::invalid_argument("Unknown option: "+k);
 }
 if(a.c.device<0||a.c.m<=0||a.c.n<=0||a.c.k<=0||a.c.inner<1||a.c.inner>65536||a.c.warmup<0||
    a.minimum_samples<1||a.seconds<=0||a.window<=0||a.warmup_seconds<0||a.c.lt_candidates<1||a.c.lt_candidates>512)
    throw std::invalid_argument("Invalid numeric range");
 if(a.c.backend!="cublas"&&a.c.backend!="cublaslt")throw std::invalid_argument("Invalid API");
 if(a.c.mode!="graph"&&a.c.mode!="launch")throw std::invalid_argument("Invalid timing");
 if(a.phase!="scan"&&a.phase!="sustained")throw std::invalid_argument("Invalid phase");
 if(!a.info&&(a.raw.empty()||a.events.empty()||a.trials.empty()))throw std::invalid_argument("raw/events/trials paths required");
 return a;
}
std::string uuid(const cudaDeviceProp& p) {
 std::ostringstream o;o<<"GPU-"<<std::hex<<std::setfill('0');
 for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)o<<'-';o<<std::setw(2)<<unsigned(static_cast<unsigned char>(p.uuid.bytes[i]));}return o.str();
}
struct Phase {
 const Args& a;std::string name,gpu;long long start=ns(),unix_start=ns(CLOCK_REALTIME);
 Phase(const Args& a_,std::string n,std::string g):a(a_),name(std::move(n)),gpu(std::move(g)){}
 ~Phase(){try{csv(a.events,{"run_id","case_id","api","gpu_uuid","phase","start_monotonic_ns","end_monotonic_ns","start_unix_ns","end_unix_ns"},
 {{"run_id",a.run},{"case_id",a.id},{"api",a.c.backend},{"gpu_uuid",gpu},{"phase",name},
 {"start_monotonic_ns",text(start)},{"end_monotonic_ns",text(ns())},{"start_unix_ns",text(unix_start)},{"end_unix_ns",text(ns(CLOCK_REALTIME))}});}catch(...){}}
};
__host__ __device__ unsigned long long mix(unsigned long long x) {
 x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
__host__ __device__ float datum(std::size_t i,unsigned long long salt) {
 return __half2float(__float2half_rn(float(mix(i^salt)>>40)*(1.0f/16777216.0f)-0.5f));
}
constexpr unsigned long long seedA=1234^0xa123ULL,seedB=1234^0xb456ULL;
__global__ void initialize(__half* a,std::size_t n,unsigned long long seed) {
 for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)a[i]=__float2half_rn(datum(i,seed));
}
__global__ void gather(const __half* c,const std::size_t* indices,float* values,std::size_t n) {
 for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)values[i]=__half2float(c[indices[i]]);
}
__global__ void finite_output(const __half* c,std::size_t n,int* bad) {
 for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
  if(!isfinite(__half2float(c[i])))atomicExch(bad,1);
}
struct Reference {
 std::vector<std::size_t> indices;std::vector<double> ref,bound;double max_error=0;
 Buffer device_indices,device_values,bad;
 Reference(const Config& c):device_indices(std::min(std::size_t(c.m)*c.n,c.check_samples)*sizeof(std::size_t)),
 device_values(std::min(std::size_t(c.m)*c.n,c.check_samples)*sizeof(float)),bad(sizeof(int)) {
  std::size_t total=std::size_t(c.m)*c.n,count=std::min(total,c.check_samples);std::set<std::size_t> pick;
  for(auto x:{std::size_t(0),std::size_t(c.m-1),total-c.m,total-1,total/2})if(pick.size()<count)pick.insert(x);
  for(std::size_t j=0;pick.size()<count;++j)pick.insert(mix(j^1234)%total);
  indices.assign(pick.begin(),pick.end());
  CUDA_CHECK(cudaMemcpy(device_indices.value,indices.data(),indices.size()*sizeof(std::size_t),cudaMemcpyHostToDevice));
  for(auto idx:indices) {
   std::size_t row=idx%c.m,col=idx/c.m;double sum=0,abs_sum=0;
   for(int k=0;k<c.k;++k) {
    std::size_t ai=c.ta?std::size_t(k)+row*c.lda():row+std::size_t(k)*c.lda();
    std::size_t bi=c.tb?col+std::size_t(k)*c.ldb():std::size_t(k)+col*c.ldb();
    double v=double(datum(ai,seedA))*datum(bi,seedB);sum+=v;abs_sum+=std::abs(v);
   }
   ref.push_back(sum);bound.push_back(0.0005+0.001*std::abs(sum)+8*1.1920928955078125e-7*abs_sum);
  }
 }
 bool check(const Config& c,const void* output,cudaStream_t s) {
  CUDA_CHECK(cudaMemsetAsync(bad.value,0,sizeof(int),s));
  finite_output<<<256,256,0,s>>>(static_cast<const __half*>(output),std::size_t(c.m)*c.n,static_cast<int*>(bad.value));
  gather<<<32,128,0,s>>>(static_cast<const __half*>(output),static_cast<const std::size_t*>(device_indices.value),static_cast<float*>(device_values.value),indices.size());
  CUDA_CHECK(cudaGetLastError());std::vector<float> got(indices.size());int nonfinite=0;
  CUDA_CHECK(cudaMemcpyAsync(got.data(),device_values.value,got.size()*sizeof(float),cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaMemcpyAsync(&nonfinite,bad.value,sizeof(int),cudaMemcpyDeviceToHost,s));CUDA_CHECK(cudaStreamSynchronize(s));
  bool ok=!nonfinite;max_error=0;for(std::size_t i=0;i<got.size();++i){double e=std::abs(got[i]-ref[i]);max_error=std::max(max_error,e);if(!std::isfinite(got[i])||e>bound[i])ok=false;}return ok;
 }
};
std::string attr(const std::string& desc,const std::string& key) {
 std::string token=key+"=";auto i=desc.find(token);if(i==std::string::npos)return "";i+=token.size();return desc.substr(i,desc.find(';',i)-i);
}
void run(const Args& a,Fields& base,const std::string& gpu) {
 const Config& c=a.c;Stream stream;
 std::size_t free=0,total=0;CUDA_CHECK(cudaMemGetInfo(&free,&total));
 long double need=2.L*(static_cast<long double>(c.m)*c.k+static_cast<long double>(c.k)*c.n+static_cast<long double>(c.m)*c.n)+c.workspace+2*(long double)c.check_samples*sizeof(double);
 if(need+std::min<std::size_t>(256ULL<<20,total/10)>free)throw Skip("Insufficient VRAM including workspace and reserve");
 Buffer A(2ULL*c.m*c.k),B(2ULL*c.k*c.n),C(2ULL*c.m*c.n);
 {
  Phase p(a,"initialization",gpu);
  initialize<<<256,256,0,stream.value>>>(static_cast<__half*>(A.value),std::size_t(c.m)*c.k,seedA);
  initialize<<<256,256,0,stream.value>>>(static_cast<__half*>(B.value),std::size_t(c.k)*c.n,seedB);
  CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaMemsetAsync(C.value,0xff,2ULL*c.m*c.n,stream.value));CUDA_CHECK(cudaStreamSynchronize(stream.value));
 }
 std::unique_ptr<Reference> reference;
 {Phase p(a,"reference",gpu);reference=std::make_unique<Reference>(c);}
 std::unique_ptr<GemmPlan> plan;
 {
  Phase p(a,"tuning_validation",gpu);
  if(c.backend=="cublas")plan=make_cublas(c,A.value,B.value,C.value,stream.value);
  else {
   int rejected=0;
   try {
    plan=make_cublaslt(c,A.value,B.value,C.value,nullptr,2ULL*c.m*c.n,stream.value,
     [&](const AlgorithmInfo& trial){
      bool ok=reference->check(c,C.value,stream.value);if(!ok)++rejected;
      csv((std::filesystem::path(a.trials).parent_path()/"candidate_validation.csv").string(),
       {"run_id","case_id","heuristic_index","algorithm","correctness_pass","max_abs_error"},
       {{"run_id",a.run},{"case_id",a.id},{"heuristic_index",text(trial.heuristic_index)},
        {"algorithm",trial.description},{"correctness_pass",ok?"true":"false"},{"max_abs_error",text(reference->max_error)}});
      return ok;
     });
   } catch(const Skip&) {
    if(rejected)throw std::runtime_error("All executable Lt candidates failed correctness; see candidate_validation.csv");
    throw;
   }
  }
  for(const auto& t:plan->info.trials)csv(a.trials,{"run_id","case_id","heuristic_index","algorithm","workspace_bytes","correctness_status","tune_median_ms"},
   {{"run_id",a.run},{"case_id",a.id},{"heuristic_index",text(t.index)},{"algorithm",t.description},{"workspace_bytes",text(t.workspace)},
    {"correctness_status",t.status},{"tune_median_ms",t.status=="VALIDATED"?text(t.median_ms):""}});
  plan->run();CUDA_CHECK(cudaStreamSynchronize(stream.value));
  if(!reference->check(c,C.value,stream.value))throw std::runtime_error("Correctness failed BEFORE performance measurement");
 }
 base["algorithm"]=plan->info.description;base["algorithm_id"]=attr(plan->info.description,"id");
 base["workspace_bytes"]=c.backend=="cublaslt"?text(plan->info.workspace_used):"";
 base["heuristic_candidates"]=c.backend=="cublaslt"?text(plan->info.candidates):"";
 auto flags=attr(plan->info.description,"numeric_flags");
 base["tensor_core_evidence"]=flags.empty()?"unverified_requires_profiler":
 ((std::stoull(flags)&CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)?"lt_numerical_flags_tensor_core":"lt_flags_no_hmma");
 base["instruction_path"]="unverified_requires_sass";
 base["correctness_pass"]="true";base["checked_elements"]=text(reference->indices.size());base["max_abs_error"]=text(reference->max_error);
 int inner=c.inner;
 {
  Phase p(a,"warmup",gpu);long long begin=ns();int rounds=0;
  do {for(int j=0;j<c.inner;++j)plan->run();CUDA_CHECK(cudaStreamSynchronize(stream.value));++rounds;}
  while(rounds<c.warmup||(ns()-begin)*1e-9<a.warmup_seconds);
  Event start,stop;CUDA_CHECK(cudaEventRecord(start.value,stream.value));for(int j=0;j<c.inner;++j)plan->run();
  CUDA_CHECK(cudaEventRecord(stop.value,stream.value));CUDA_CHECK(cudaEventSynchronize(stop.value));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));
  inner=std::clamp(int(20.0/std::max(double(ms)/c.inner,0.0001)),1,4096);
 }
 Graph graph;
 {
  Phase p(a,"timing_setup",gpu);
  if(c.mode=="graph") {
   CUDA_CHECK(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
   for(int j=0;j<inner;++j)plan->run();
   CUDA_CHECK(cudaStreamEndCapture(stream.value,&graph.graph));CUDA_CHECK(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
   CUDA_CHECK(cudaGraphLaunch(graph.executable,stream.value));CUDA_CHECK(cudaStreamSynchronize(stream.value));
  }
 }
 auto batch=[&](){if(c.mode=="graph")CUDA_CHECK(cudaGraphLaunch(graph.executable,stream.value));else for(int j=0;j<inner;++j)plan->run();};
 if(a.profile)CUDA_CHECK(cudaProfilerStart());
 Event start,stop;long long run_start=ns();int total_samples=0,window=0;
 {
  Phase phase(a,a.phase,gpu);
  do {
   long long begin=ns(),utc=ns(CLOCK_REALTIME);std::vector<double> samples;
   do {
    CUDA_CHECK(cudaEventRecord(start.value,stream.value));batch();CUDA_CHECK(cudaEventRecord(stop.value,stream.value));
    CUDA_CHECK(cudaEventSynchronize(stop.value));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,start.value,stop.value));
    if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("Invalid CUDA event time");
    samples.push_back(ms/inner);++total_samples;
   } while((ns()-begin)*1e-9<a.window && ((ns()-run_start)*1e-9<a.seconds||total_samples<a.minimum_samples));
   long long end=ns(),utc_end=ns(CLOCK_REALTIME);auto sorted=samples;std::sort(sorted.begin(),sorted.end());
   double mean=std::accumulate(samples.begin(),samples.end(),0.0)/samples.size(),median=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*0.5;
   Fields row=base;std::ostringstream list;list<<std::setprecision(15);for(std::size_t i=0;i<samples.size();++i){if(i)list<<';';list<<samples[i];}
   row["window"]=text(window++);row["start_monotonic_ns"]=text(begin);row["end_monotonic_ns"]=text(end);
   row["start_unix_ns"]=text(utc);row["end_unix_ns"]=text(utc_end);
   row["elapsed_ms_median"]=text(median);row["elapsed_ms_mean"]=text(mean);row["elapsed_ms_min"]=text(sorted.front());
   row["tflops"]=text(2.0*c.m*c.n*c.k/(median*1e9));row["tflops_mean"]=text(2.0*c.m*c.n*c.k/(mean*1e9));
   row["inner"]=text(inner);row["repeats"]=text(samples.size());row["gpu_total_ms"]=text(mean*samples.size()*inner);
   row["wall_ms"]=text((end-begin)*1e-6);row["samples_ms"]=list.str();row["status"]="PASS";csv(a.raw,raw_header,row);
  } while((ns()-run_start)*1e-9<a.seconds||total_samples<a.minimum_samples);
 }
 {
  if(a.profile)CUDA_CHECK(cudaProfilerStop());
  Phase p(a,"post_validation",gpu);
  if(!reference->check(c,C.value,stream.value))throw std::runtime_error("Correctness failed AFTER benchmark; invalidate entire run_id");
 }
}
int main(int argc,char** argv) {
 Args a;Fields base;
 try {
  a=args(argc,argv);base={{"run_id",a.run},{"case_id",a.id},{"phase",a.phase},{"M",text(a.c.m)},{"N",text(a.c.n)},{"K",text(a.c.k)},
  {"transA",a.c.ta?"T":"N"},{"transB",a.c.tb?"T":"N"},{"dtype","FP16"},{"compute_type","FP32"},{"api",a.c.backend}};
  CUDA_CHECK(cudaSetDevice(a.c.device));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,a.c.device));base["gpu_uuid"]=uuid(prop);
  if(a.info){int runtime=0,driver=0,blas=0;CUDA_CHECK(cudaRuntimeGetVersion(&runtime));CUDA_CHECK(cudaDriverGetVersion(&driver));BlasHandle h;BLAS_CHECK(cublasGetVersion(h.value,&blas));
   std::cout<<"{\"gpu_name\":"<<json_string(prop.name)<<",\"gpu_uuid\":"<<json_string(uuid(prop))<<",\"sm_count\":"<<prop.multiProcessorCount
   <<",\"compute_capability\":"<<json_string(text(prop.major)+"."+text(prop.minor))<<",\"vram_bytes\":"<<prop.totalGlobalMem
   <<",\"cuda_runtime\":"<<runtime<<",\"cuda_driver_api\":"<<driver<<",\"cublas_version\":"<<blas<<"}\n";return 0;}
  run(a,base,base["gpu_uuid"]);std::cout<<"PASS "<<a.run<<"\n";return 0;
 } catch(const Skip& e){base["status"]="SKIP";base["reason"]=e.what();if(!a.raw.empty())csv(a.raw,raw_header,base);std::cerr<<e.what()<<'\n';return 3;}
 catch(const std::exception& e){base["status"]="FAIL";base["reason"]=e.what();base["correctness_pass"]="false";if(!a.raw.empty())csv(a.raw,raw_header,base);std::cerr<<e.what()<<'\n';return 1;}
}
