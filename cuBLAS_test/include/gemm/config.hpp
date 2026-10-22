#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace gemm {
enum class Dtype { FP16, BF16, FP32, TF32 };
struct Config {
  int m=1024, n=1024, k=1024;
  bool ta=false, tb=false;
  float alpha=1, beta=0;
  Dtype dtype=Dtype::FP16;
  bool output_fp32=false;
  std::string backend="cublas", mode="launch", layout="row_major", suite, csv="results/benchmark.csv";
  int device=0, warmup=10, repeats=30, inner=10;
  std::uint64_t seed=1234;
  std::size_t check_samples=512;
  bool check_all=false, help=false, device_info=false;
  double atol=-1, rtol=-1;
  std::size_t workspace=64ull*1024*1024;
  std::string lt_selection="first";
  int lt_candidates=32, lt_index=0;
  int lda() const { return layout=="column_major" ? (ta ? k : m) : (ta ? m : k); }
  int ldb() const { return layout=="column_major" ? (tb ? n : k) : (tb ? k : n); }
  int ldc() const { return layout=="column_major" ? m : n; }
  int effective_inner() const { return beta == 0 ? inner : 1; }
};
std::string dtype_name(Dtype);
std::string output_name(const Config&);
std::string compute_name(const Config&);
Config parse_args(int argc, char** argv);
std::vector<Config> cases(const Config&);
void print_help();
}

