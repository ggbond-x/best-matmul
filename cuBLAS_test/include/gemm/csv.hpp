#pragma once
#include "gemm/config.hpp"

#include <cstddef>
#include <string>

namespace gemm {

// 每轮真实测量一行。summary.csv 由 summarize.py 从该文件聚合得到。
struct RawRow {
  std::string plan_id;
  std::string run_id;
  std::string case_id;
  std::string shape_id;
  int m = 0;
  int n = 0;
  int k = 0;
  std::string layout;

  std::string a_dtype = "FP16";
  std::string b_dtype = "FP16";
  std::string accum_dtype;
  std::string out_dtype = "FP16";
  double alpha = 1.0;
  double beta = 0.0;

  std::string a_storage;
  std::string b_storage;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;

  std::string implementation;
  std::string kernel_name;
  std::string algorithm_id;
  std::size_t workspace_bytes = 0;

  std::string timing_mode;
  std::string timer_scope;
  std::string cache_mode;
  int round = 0;
  int samples = 0;
  int inner = 0;
  std::string round_started_utc;
  std::string round_finished_utc;
  std::string telemetry_scope;

  double duration_us_median = 0.0;
  double duration_us_p10 = 0.0;
  double duration_us_p90 = 0.0;
  double duration_us_min = 0.0;
  double duration_us_max = 0.0;
  double tflops_median = 0.0;

  double sm_clock_mhz_avg = 0.0;
  double mem_clock_mhz_avg = 0.0;
  double power_w_avg = 0.0;
  double power_w_max = 0.0;
  double temp_c_max = 0.0;
  bool clock_locked = false;
  int clock_target_mhz = 0;

  bool correctness_checked = false;
  bool correctness_pass = false;
  std::string check_mode;

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double rel_l2_error = 0.0;
  std::size_t mismatch_count = 0;
  std::size_t nan_inf_count = 0;
  std::size_t checked_elements = 0;

  std::string gpu_name;
  std::string gpu_uuid;
  int sm_count = 0;
  std::string compute_capability;

  std::string build_git_sha;
  std::string config_hash;
  std::string status;
  std::string notes;
};

void append_raw_row(std::string const& path, RawRow const& row);

// 仅用于 ncu_summary.csv；没有采集到报告时写 FAILED 行，不编造指标。
struct NcuRow {
  std::string plan_id;
  std::string run_id;
  std::string case_id;
  std::string implementation;
  std::string kernel_name;
  std::string launch_config;
  std::string report_path;
  std::string status;
  std::string notes;
};

void append_ncu_row(std::string const& path, NcuRow const& row);

}  // namespace gemm
