#include "gemm/csv.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gemm {
namespace {

std::string number(double value) {
  if (!std::isfinite(value)) return "N/A";
  std::ostringstream stream;
  stream << std::setprecision(10) << value;
  return stream.str();
}

std::string integer(std::size_t value) { return std::to_string(value); }

std::string boolean(bool value) { return value ? "true" : "false"; }

using Row = std::vector<std::pair<std::string, std::string>>;

std::string csv_line(Row const& row, bool header) {
  std::string line;
  for (auto const& field : row) {
    if (!line.empty()) line += ',';
    line += quote_csv(header ? field.first : field.second);
  }
  return line;
}

void append_row(std::string const& path, Row const& row) {
  if (path.empty()) return;
  std::filesystem::path p(path);
  if (!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path());
  bool const exists = std::filesystem::exists(p) && std::filesystem::file_size(p) > 0;
  if (exists) {
    std::ifstream input(p);
    std::string header;
    std::getline(input, header);
    if (!header.empty() && header.back() == '\r') header.pop_back();
    if (header != csv_line(row, true)) {
      throw std::runtime_error("CSV schema mismatch; refusing to append to " + path);
    }
  }
  std::ofstream output(p, std::ios::app);
  if (!output) throw std::runtime_error("Cannot open CSV for append: " + path);
  if (!exists) output << csv_line(row, true) << '\n';
  output << csv_line(row, false) << '\n';
  output.flush();
  if (!output) throw std::runtime_error("Cannot write CSV: " + path);
}

}  // namespace

void append_raw_row(std::string const& path, RawRow const& r) {
  Row row = {
      {"plan_id", r.plan_id},
      {"run_id", r.run_id},
      {"case_id", r.case_id},
      {"shape_id", r.shape_id},
      {"m", std::to_string(r.m)},
      {"n", std::to_string(r.n)},
      {"k", std::to_string(r.k)},
      {"layout", r.layout},
      {"a_dtype", r.a_dtype},
      {"b_dtype", r.b_dtype},
      {"accum_dtype", r.accum_dtype},
      {"out_dtype", r.out_dtype},
      {"alpha", number(r.alpha)},
      {"beta", number(r.beta)},
      {"a_storage", r.a_storage},
      {"b_storage", r.b_storage},
      {"lda", std::to_string(r.lda)},
      {"ldb", std::to_string(r.ldb)},
      {"ldc", std::to_string(r.ldc)},
      {"implementation", r.implementation},
      {"kernel_name", r.kernel_name},
      {"algorithm_id", r.algorithm_id},
      {"workspace_bytes", integer(r.workspace_bytes)},
      {"timing_mode", r.timing_mode},
      {"timer_scope", r.timer_scope},
      {"cache_mode", r.cache_mode},
      {"round", std::to_string(r.round)},
      {"samples", std::to_string(r.samples)},
      {"inner", std::to_string(r.inner)},
      {"round_started_utc", r.round_started_utc},
      {"round_finished_utc", r.round_finished_utc},
      {"telemetry_scope", r.telemetry_scope},
      {"duration_us_median", number(r.duration_us_median)},
      {"duration_us_p10", number(r.duration_us_p10)},
      {"duration_us_p90", number(r.duration_us_p90)},
      {"duration_us_min", number(r.duration_us_min)},
      {"duration_us_max", number(r.duration_us_max)},
      {"tflops_median", number(r.tflops_median)},
      {"sm_clock_mhz_avg", number(r.sm_clock_mhz_avg)},
      {"mem_clock_mhz_avg", number(r.mem_clock_mhz_avg)},
      {"power_w_avg", number(r.power_w_avg)},
      {"power_w_max", number(r.power_w_max)},
      {"temp_c_max", number(r.temp_c_max)},
      {"clock_locked", boolean(r.clock_locked)},
      {"clock_target_mhz", std::to_string(r.clock_target_mhz)},
      {"correctness_checked", boolean(r.correctness_checked)},
      {"correctness_pass", r.correctness_checked ? (r.correctness_pass ? "PASS" : "FAIL")
                                                  : "NOT_CHECKED"},
      {"check_mode", r.check_mode},
      {"max_abs_error", number(r.max_abs_error)},
      {"max_rel_error", number(r.max_rel_error)},
      {"rel_l2_error", number(r.rel_l2_error)},
      {"mismatch_count", integer(r.mismatch_count)},
      {"nan_inf_count", integer(r.nan_inf_count)},
      {"checked_elements", integer(r.checked_elements)},
      {"gpu_name", r.gpu_name},
      {"gpu_uuid", r.gpu_uuid},
      {"sm_count", std::to_string(r.sm_count)},
      {"compute_capability", r.compute_capability},
      {"build_git_sha", r.build_git_sha},
      {"config_hash", r.config_hash},
      {"status", r.status},
      {"notes", r.notes},
  };
  append_row(path, row);
}

void append_ncu_row(std::string const& path, NcuRow const& r) {
  Row row = {
      {"plan_id", r.plan_id},       {"run_id", r.run_id},
      {"case_id", r.case_id},       {"implementation", r.implementation},
      {"kernel_name", r.kernel_name},
      {"launch_config", r.launch_config},
      {"report_path", r.report_path},
      {"status", r.status},         {"notes", r.notes},
  };
  append_row(path, row);
}

}  // namespace gemm
