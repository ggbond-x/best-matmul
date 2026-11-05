// plan1 统一基准入口。
//
// 一个进程内按“case -> cache -> layout -> implementation -> round”执行：
//   1. 生成固定输入并计算 FP64 参考（一次性）；
//   2. 每个实现先做正确性验证，未通过不进入计时；
//   3. 逐轮计时；同一 case 内每轮轮换实现顺序，避免热/频漂移系统性偏向某个实现。
// 所有真实测量立即写入 csvs/raw_runs.csv，失败也保留已完成的数据。
#include "gemm/backend.hpp"
#include "gemm/bench.hpp"
#include "gemm/config.hpp"
#include "gemm/csv.hpp"
#include "gemm/cuda_utils.hpp"
#include "gemm/telemetry.hpp"
#include "gemm/timing.hpp"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cutlass/version.h>
#include <nvml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using namespace gemm;

double const kNaN = std::numeric_limits<double>::quiet_NaN();

struct CaseDef {
  std::string id;
  std::string shape_id;
  int m = 0;
  int n = 0;
  int k = 0;
  std::string priority;
  bool configured_thresholds = false;
  Thresholds fp16;
  Thresholds fp32;
  CaseDef() = default;
  CaseDef(std::string id_value, std::string shape_value, int m_value, int n_value,
          int k_value, std::string priority_value)
      : id(id_value), shape_id(shape_value), m(m_value), n(n_value), k(k_value),
        priority(priority_value) {}
  Thresholds limits(Accum accum) const {
    return configured_thresholds ? (accum == Accum::FP16 ? fp16 : fp32) : thresholds_for(accum, k);
  }
};

struct GpuInfo {
  std::string name;
  std::string uuid;
  int sm_count = 0;
  int major = 0;
  int minor = 0;
  std::size_t total_memory = 0;
};

struct ImplOutcome {
  std::string implementation;
  std::string layout;
  std::string cache_mode;
  std::string status;
  std::string reason;
  int rows = 0;
  double duration_us_median = kNaN;
  double tflops_median = kNaN;
};

struct CaseOutcome {
  CaseDef def;
  std::vector<ImplOutcome> implementations;
};

std::string now_utc() {
  std::time_t const t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buffer;
}

std::string hostname() {
  char buffer[256] = {0};
  if (gethostname(buffer, sizeof(buffer) - 1) != 0) return "unknown";
  return buffer;
}

std::vector<std::string> split_list(std::string const& text) {
  std::vector<std::string> parts;
  std::string current;
  for (char ch : text) {
    if (ch == ',' || ch == ';') {
      if (!current.empty()) parts.push_back(current);
      current.clear();
    } else if (!std::isspace(static_cast<unsigned char>(ch))) {
      current += ch;
    }
  }
  if (!current.empty()) parts.push_back(current);
  return parts;
}

std::vector<CaseDef> default_cases() {
  return {
      {"S01", "S01", 4096, 4096, 4096, "P0"},
      {"S02", "S02", 4096, 4096, 256, "P0"},
      {"S03", "S03", 4096, 4096, 1024, "P0"},
      {"S04", "S04", 4096, 4096, 2048, "P0"},
      {"S05", "S05", 2048, 2048, 2048, "P0"},
      {"S06", "S06", 8192, 8192, 8192, "P0"},
      {"S07", "S07", 1024, 1024, 1024, "P1"},
      {"S08", "S08", 4096, 4096, 512, "P1"},
      {"S09", "S09", 256, 4096, 4096, "P1"},
      {"S10", "S10", 4096, 256, 4096, "P1"},
      {"S11", "S11", 1024, 8192, 4096, "P1"},
      {"S12", "S12", 8192, 1024, 4096, "P1"},
  };
}

void usage() {
  std::printf(
      "Usage: gemm_bench [options]\n"
      "  --mode quick|full|ncu        run label (default: quick)\n"
      "  --stage bench|ncu            bench = correctness+timing, ncu = one launch (default: bench)\n"
      "  --cases S01,S02              cases to run (default: quick S01,S02; full all; ncu S01)\n"
      "  --case-defs CSV              resolved ID:M:N:K:priority and per-case FP16/FP32 thresholds\n"
      "  --cold-cases S01,S02         additionally measure these cases with cold-cache timing\n"
      "  --layouts TN,NT              layouts (default: TN,NT)\n"
      "  --impl NAME[,NAME]           implementations (default: all registered)\n"
      "  --timing event|wall|graph    timing mode (default: event)\n"
      "  --cache warm|cold            cache mode (default: warm)\n"
      "  --warmup N --inner N --samples N --rounds N\n"
      "  --flush-mb N                 L2 flush buffer size for cold mode (default: 256)\n"
      "  --seed N                     input seed (default: 20261105)\n"
      "  --lt-workspace-mb N --lt-candidates N --lt-selection tune|first|index --lt-index N\n"
      "  --skip-reference             do not build FP64 reference / correctness (diagnostic)\n"
      "  --skip-timing                only run correctness\n"
      "  --csv PATH --manifest PATH --correctness-log PATH --environment-json PATH\n"
      "  --plan-id ID --run-id ID --config-hash HASH --git-sha SHA\n"
      "  --clock-locked true|false --clock-target MHz --device N --notes TEXT\n"
      "  --list-impls --list-cases --help\n");
}

struct Cli {
  Options options;
  std::string stage = "bench";
  std::vector<std::string> cases;
  std::vector<CaseDef> case_defs;
  std::vector<std::string> cold_cases;
  std::vector<std::string> layouts;
  std::vector<std::string> impls;
  std::string environment_json;
  bool list_impls = false;
  bool list_cases = false;
  std::string command_line;
};

std::string join(int argc, char** argv) {
  std::string out;
  for (int i = 0; i < argc; ++i) {
    if (i) out += ' ';
    out += argv[i];
  }
  return out;
}

int parse_int(std::string const& text, char const* name) {
  try {
    std::size_t used = 0;
    long const value = std::stol(text, &used);
    if (used != text.size() || value < 0 || value > 1000000000L) {
      throw std::invalid_argument("range");
    }
    return int(value);
  } catch (std::exception const&) {
    throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + text);
  }
}

double parse_double(std::string const& text, char const* name) {
  try {
    std::size_t used = 0;
    double const value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("range");
    }
    return value;
  } catch (std::exception const&) {
    throw std::invalid_argument(std::string("invalid number for ") + name + ": " + text);
  }
}

bool parse_bool(std::string const& text) {
  return text == "1" || text == "true" || text == "yes" || text == "on";
}

Cli parse_cli(int argc, char** argv) {
  Cli cli;
  cli.command_line = join(argc, argv);
  cli.options.mode = "quick";
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    auto need = [&](char const* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage();
      std::exit(0);
    } else if (arg == "--mode") {
      cli.options.mode = need("--mode");
    } else if (arg == "--stage") {
      cli.stage = need("--stage");
    } else if (arg == "--cases") {
      for (auto const& v : split_list(need("--cases"))) cli.cases.push_back(v);
    } else if (arg == "--case-defs") {
      for (auto const& entry : split_list(need("--case-defs"))) {
        std::vector<std::string> fields;
        std::stringstream stream(entry);
        std::string field;
        while (std::getline(stream, field, ':')) fields.push_back(field);
        if (fields.size() != 13 || fields[0].empty() || fields[4].empty())
          throw std::invalid_argument("invalid --case-defs entry: " + entry);
        CaseDef c{fields[0], fields[0], parse_int(fields[1], "M"), parse_int(fields[2], "N"),
                  parse_int(fields[3], "K"), fields[4]};
        if (!c.m || !c.n || !c.k) throw std::invalid_argument("case dimensions must be positive");
        auto limits = [&](int i) {
          Thresholds t{parse_double(fields[i], "atol"), parse_double(fields[i+1], "rtol"),
                       parse_double(fields[i+2], "rel_l2_max"), parse_double(fields[i+3], "mismatch_fraction_max")};
          if (t.atol < 0 || t.rtol < 0 || t.rel_l2_max < 0 || t.mismatch_fraction_max < 0 || t.mismatch_fraction_max > 1)
            throw std::invalid_argument("invalid correctness thresholds");
          return t;
        };
        c.configured_thresholds = true;
        c.fp16 = limits(5);
        c.fp32 = limits(9);
        cli.case_defs.push_back(c);
      }
    } else if (arg == "--cold-cases") {
      for (auto const& v : split_list(need("--cold-cases"))) cli.cold_cases.push_back(v);
    } else if (arg == "--layouts" || arg == "--layout") {
      for (auto const& v : split_list(need("--layouts"))) cli.layouts.push_back(v);
    } else if (arg == "--impl" || arg == "--implementation") {
      for (auto const& v : split_list(need("--impl"))) cli.impls.push_back(v);
    } else if (arg == "--timing") {
      cli.options.timing_mode = need("--timing");
    } else if (arg == "--cache") {
      cli.options.cache_mode = need("--cache");
    } else if (arg == "--warmup") {
      cli.options.warmup = parse_int(need("--warmup"), "--warmup");
    } else if (arg == "--inner") {
      cli.options.inner = parse_int(need("--inner"), "--inner");
    } else if (arg == "--samples") {
      cli.options.samples = parse_int(need("--samples"), "--samples");
    } else if (arg == "--rounds") {
      cli.options.rounds = parse_int(need("--rounds"), "--rounds");
    } else if (arg == "--flush-mb") {
      cli.options.flush_bytes = std::size_t(parse_int(need("--flush-mb"), "--flush-mb")) << 20;
    } else if (arg == "--seed") {
      cli.options.seed = std::uint64_t(parse_int(need("--seed"), "--seed"));
    } else if (arg == "--lt-workspace-mb") {
      cli.options.lt_workspace =
          std::size_t(parse_int(need("--lt-workspace-mb"), "--lt-workspace-mb")) << 20;
    } else if (arg == "--lt-candidates") {
      cli.options.lt_candidates = parse_int(need("--lt-candidates"), "--lt-candidates");
    } else if (arg == "--lt-selection") {
      cli.options.lt_selection = need("--lt-selection");
    } else if (arg == "--lt-index") {
      cli.options.lt_index = parse_int(need("--lt-index"), "--lt-index");
    } else if (arg == "--lt-algo-hex") {
      cli.options.lt_algo_hex = need("--lt-algo-hex");
    } else if (arg == "--lt-algo-version") {
      cli.options.lt_algo_version = need("--lt-algo-version");
    } else if (arg == "--skip-reference") {
      cli.options.compute_reference = false;
      cli.options.validate = false;
    } else if (arg == "--skip-timing") {
      cli.options.skip_timing = true;
    } else if (arg == "--csv") {
      cli.options.csv_path = need("--csv");
    } else if (arg == "--manifest") {
      cli.options.manifest_path = need("--manifest");
    } else if (arg == "--correctness-log") {
      cli.options.correctness_log_path = need("--correctness-log");
    } else if (arg == "--environment-json") {
      cli.environment_json = need("--environment-json");
    } else if (arg == "--plan-id") {
      cli.options.plan_id = need("--plan-id");
    } else if (arg == "--run-id") {
      cli.options.run_id = need("--run-id");
    } else if (arg == "--config-hash") {
      cli.options.config_hash = need("--config-hash");
    } else if (arg == "--git-sha") {
      cli.options.build_git_sha = need("--git-sha");
    } else if (arg == "--clock-locked") {
      cli.options.clock_locked = parse_bool(need("--clock-locked"));
    } else if (arg == "--clock-target") {
      cli.options.clock_target_mhz = parse_int(need("--clock-target"), "--clock-target");
    } else if (arg == "--device") {
      cli.options.device = parse_int(need("--device"), "--device");
    } else if (arg == "--notes") {
      cli.options.notes = need("--notes");
    } else if (arg == "--alpha") {
      cli.options.alpha = float(parse_double(need("--alpha"), "--alpha"));
    } else if (arg == "--beta") {
      cli.options.beta = float(parse_double(need("--beta"), "--beta"));
    } else if (arg == "--list-impls") {
      cli.list_impls = true;
    } else if (arg == "--list-cases") {
      cli.list_cases = true;
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }

  if (cli.options.mode != "quick" && cli.options.mode != "full" &&
      cli.options.mode != "ncu") {
    throw std::invalid_argument("--mode must be quick, full or ncu");
  }
  if (cli.stage != "bench" && cli.stage != "ncu") {
    throw std::invalid_argument("--stage must be bench or ncu");
  }
  if (cli.options.timing_mode != "event" && cli.options.timing_mode != "wall" &&
      cli.options.timing_mode != "graph") {
    throw std::invalid_argument("--timing must be event, wall or graph");
  }
  if (cli.options.cache_mode != "warm" && cli.options.cache_mode != "cold") {
    throw std::invalid_argument("--cache must be warm or cold");
  }
  if (cli.options.alpha != 1.0f || cli.options.beta != 0.0f) {
    throw std::invalid_argument(
        "plan1 only defines D = 1.0 * A * B^T + 0.0 * C; --alpha/--beta must be 1 and 0");
  }
  if (cli.options.warmup < 0 || cli.options.inner < 1 || cli.options.samples < 1 ||
      cli.options.rounds < 1) {
    throw std::invalid_argument("require warmup>=0, inner>=1, samples>=1, rounds>=1");
  }
  if (cli.layouts.empty()) cli.layouts = {"TN", "NT"};
  for (auto const& layout : cli.layouts) {
    if (layout != "TN" && layout != "NT") {
      throw std::invalid_argument("layout must be TN or NT: " + layout);
    }
  }
  if (cli.options.lt_selection != "tune" && cli.options.lt_selection != "first" &&
      cli.options.lt_selection != "index") {
    throw std::invalid_argument("--lt-selection must be tune, first or index");
  }
  if (cli.stage == "ncu" && cli.options.lt_algo_hex.empty()) cli.options.lt_selection = "first";
  return cli;
}

std::string uuid_string(cudaUUID_t const& uuid) {
  // cudaUUID_t::bytes 是 char（可能带符号）；用 hhx 避免符号扩展成 ffffffxx。
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer),
                "GPU-%02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-"
                "%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx",
                uuid.bytes[0], uuid.bytes[1], uuid.bytes[2], uuid.bytes[3], uuid.bytes[4],
                uuid.bytes[5], uuid.bytes[6], uuid.bytes[7], uuid.bytes[8], uuid.bytes[9],
                uuid.bytes[10], uuid.bytes[11], uuid.bytes[12], uuid.bytes[13], uuid.bytes[14],
                uuid.bytes[15]);
  return buffer;
}

std::string pci_bus_string(cudaDeviceProp const& prop) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID,
                prop.pciDeviceID);
  return buffer;
}

GpuInfo gpu_info(cudaDeviceProp const& prop) {
  GpuInfo info;
  info.name = prop.name;
  info.uuid = uuid_string(prop.uuid);
  info.sm_count = prop.multiProcessorCount;
  info.major = prop.major;
  info.minor = prop.minor;
  info.total_memory = prop.totalGlobalMem;
  return info;
}

std::string cublas_version_text() {
  int major = 0;
  int minor = 0;
  int patch = 0;
  (void)cublasGetProperty(MAJOR_VERSION, &major);
  (void)cublasGetProperty(MINOR_VERSION, &minor);
  (void)cublasGetProperty(PATCH_LEVEL, &patch);
  std::ostringstream stream;
  stream << major << "." << minor << "." << patch;
  return stream.str();
}

std::string cuda_version_text(int version) {
  std::ostringstream stream;
  stream << version / 1000 << "." << (version % 1000) / 10;
  return stream.str();
}

void write_environment_json(Cli const& cli, GpuInfo const& gpu, cudaDeviceProp const& prop,
                            TelemetryReader const& telemetry, std::string const& started) {
  if (cli.environment_json.empty()) return;
  int runtime = 0;
  int driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  TelemetrySample const sample = telemetry.sample();

  std::ofstream out(cli.environment_json);
  if (!out) throw std::runtime_error("cannot write environment json: " + cli.environment_json);
  out << "{\n";
  out << "  \"plan_id\": \"" << json_escape(cli.options.plan_id) << "\",\n";
  out << "  \"run_id\": \"" << json_escape(cli.options.run_id) << "\",\n";
  out << "  \"mode\": \"" << json_escape(cli.options.mode) << "\",\n";
  out << "  \"stage\": \"" << json_escape(cli.stage) << "\",\n";
  out << "  \"started_utc\": \"" << started << "\",\n";
  out << "  \"hostname\": \"" << json_escape(hostname()) << "\",\n";
  out << "  \"command_line\": \"" << json_escape(cli.command_line) << "\",\n";
  out << "  \"gpu\": {\"name\": \"" << json_escape(gpu.name) << "\", \"uuid\": \""
      << gpu.uuid << "\", \"sm_count\": " << gpu.sm_count << ", \"compute_capability\": \""
      << gpu.major << "." << gpu.minor << "\", \"total_memory_bytes\": " << gpu.total_memory
      << ", \"pci_bus_id\": \"" << pci_bus_string(prop) << "\"},\n";
  out << "  \"cuda_driver_version\": \"" << cuda_version_text(driver) << "\",\n";
  out << "  \"cuda_runtime_version\": \"" << cuda_version_text(runtime) << "\",\n";
  out << "  \"cublas_version\": \"" << cublas_version_text() << "\",\n";
  out << "  \"cublaslt_version\": \"" << cublasLtGetVersion() << "\",\n";
  out << "  \"cutlass_version\": \"" << CUTLASS_MAJOR << "." << CUTLASS_MINOR << "."
      << CUTLASS_PATCH << "\",\n";
  out << "  \"build_git_sha\": \"" << json_escape(cli.options.build_git_sha) << "\",\n";
  out << "  \"config_hash\": \"" << json_escape(cli.options.config_hash) << "\",\n";
  out << "  \"clock_locked\": " << (cli.options.clock_locked ? "true" : "false") << ",\n";
  out << "  \"clock_target_mhz\": " << cli.options.clock_target_mhz << ",\n";
  out << "  \"telemetry_available\": " << (telemetry.available() ? "true" : "false") << ",\n";
  if (telemetry.available()) {
    auto json_number = [](double value) { return std::isfinite(value) ? std::to_string(value) : std::string("null"); };
    out << "  \"observed_sm_clock_mhz\": " << json_number(sample.sm_clock_mhz) << ",\n";
    out << "  \"observed_mem_clock_mhz\": " << json_number(sample.mem_clock_mhz) << ",\n";
    out << "  \"observed_power_w\": " << json_number(sample.power_w) << ",\n";
    out << "  \"observed_temperature_c\": " << json_number(sample.temperature_c) << ",\n";
  } else {
    out << "  \"telemetry_error\": \"" << json_escape(telemetry.error()) << "\",\n";
  }
  out << "  \"notes\": \"" << json_escape(cli.options.notes) << "\"\n";
  out << "}\n";
}

void write_manifest(Cli const& cli, GpuInfo const& gpu, std::string const& started,
                    std::string const& overall, std::vector<CaseOutcome> const& outcomes) {
  if (cli.options.manifest_path.empty()) return;
  std::ofstream out(cli.options.manifest_path);
  if (!out) throw std::runtime_error("cannot write manifest: " + cli.options.manifest_path);
  out << "{\n";
  out << "  \"plan_id\": \"" << json_escape(cli.options.plan_id) << "\",\n";
  out << "  \"run_id\": \"" << json_escape(cli.options.run_id) << "\",\n";
  out << "  \"mode\": \"" << json_escape(cli.options.mode) << "\",\n";
  out << "  \"stage\": \"" << json_escape(cli.stage) << "\",\n";
  out << "  \"status\": \"" << overall << "\",\n";
  out << "  \"started_utc\": \"" << started << "\",\n";
  out << "  \"finished_utc\": \"" << now_utc() << "\",\n";
  out << "  \"gpu\": {\"name\": \"" << json_escape(gpu.name) << "\", \"uuid\": \"" << gpu.uuid
      << "\", \"sm_count\": " << gpu.sm_count << ", \"compute_capability\": \"" << gpu.major
      << "." << gpu.minor << "\"},\n";
  out << "  \"timing_mode\": \"" << json_escape(cli.options.timing_mode) << "\",\n";
  out << "  \"cache_mode\": \"" << json_escape(cli.options.cache_mode) << "\",\n";
  out << "  \"command_line\": \"" << json_escape(cli.command_line) << "\",\n";
  out << "  \"cases\": [\n";
  for (std::size_t ci = 0; ci < outcomes.size(); ++ci) {
    CaseOutcome const& c = outcomes[ci];
    out << "    {\"case_id\": \"" << c.def.id << "\", \"shape_id\": \"" << c.def.shape_id
        << "\", \"m\": " << c.def.m << ", \"n\": " << c.def.n << ", \"k\": " << c.def.k
        << ", \"priority\": \"" << c.def.priority << "\", \"implementations\": [\n";
    for (std::size_t ii = 0; ii < c.implementations.size(); ++ii) {
      ImplOutcome const& i = c.implementations[ii];
      out << "      {\"implementation\": \"" << json_escape(i.implementation)
          << "\", \"layout\": \"" << i.layout << "\", \"cache_mode\": \"" << i.cache_mode
          << "\", \"status\": \"" << i.status << "\", \"reason\": \"" << json_escape(i.reason)
          << "\", \"csv_rows\": " << i.rows;
      if (std::isfinite(i.duration_us_median)) {
        out << ", \"duration_us_median\": " << i.duration_us_median;
      }
      if (std::isfinite(i.tflops_median)) out << ", \"tflops_median\": " << i.tflops_median;
      out << "}" << (ii + 1 == c.implementations.size() ? "\n" : ",\n");
    }
    out << "    ]}" << (ci + 1 == outcomes.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

void correctness_log(Options const& options, CaseDef const& c, std::string const& layout,
                     std::string const& impl, Correctness const& check,
                     std::string const& status, std::string const& reason) {
  if (options.correctness_log_path.empty()) return;
  std::ofstream out(options.correctness_log_path, std::ios::app);
  if (!out) return;
  out << now_utc() << " case=" << c.id << " shape=" << c.m << "x" << c.n << "x" << c.k
      << " layout=" << layout << " implementation=" << impl << " status=" << status
      << " check_mode=" << check.mode << " checked=" << check.checked_elements
      << " max_abs=" << (std::isfinite(check.max_abs) ? check.max_abs : -1.0)
      << " max_rel=" << (std::isfinite(check.max_rel) ? check.max_rel : -1.0)
      << " rel_l2=" << (std::isfinite(check.rel_l2) ? check.rel_l2 : -1.0)
      << " mismatch=" << check.mismatch_count << " nan_inf=" << check.nonfinite_count
      << " pass=" << (check.pass ? "PASS" : "FAIL");
  if (!reason.empty()) out << " reason=\"" << reason << "\"";
  out << "\n";
}

std::string storage_a(Problem const& p) {
  return p.ta ? "row_major(K-contiguous)" : "column_major(M-contiguous)";
}
std::string storage_b(Problem const& p) {
  return p.tb ? "column_major(N-contiguous)" : "row_major(K-contiguous)";
}

RawRow make_row(Options const& options, CaseDef const& c, Problem const& p, PlanInfo const& info,
                GpuInfo const& gpu, std::string const& cache_mode, std::string const& layout,
                std::string const& timer_scope, int round, int samples, int inner) {
  RawRow row;
  row.plan_id = options.plan_id;
  row.run_id = options.run_id;
  row.case_id = c.id;
  row.shape_id = c.shape_id;
  row.m = p.m;
  row.n = p.n;
  row.k = p.k;
  row.layout = layout;
  row.accum_dtype = accum_name(info.accum);
  row.alpha = options.alpha;
  row.beta = options.beta;
  row.a_storage = storage_a(p);
  row.b_storage = storage_b(p);
  row.lda = p.ldA();
  row.ldb = p.ldB();
  row.ldc = p.ldC();
  row.implementation = info.implementation;
  row.kernel_name = info.kernel_name;
  row.algorithm_id = info.algorithm_id;
  row.workspace_bytes = info.workspace_bytes;
  row.timing_mode = options.timing_mode;
  row.timer_scope = timer_scope;
  row.cache_mode = cache_mode;
  row.round = round;
  row.samples = samples;
  row.inner = inner;
  row.duration_us_median = kNaN;
  row.duration_us_p10 = kNaN;
  row.duration_us_p90 = kNaN;
  row.duration_us_min = kNaN;
  row.duration_us_max = kNaN;
  row.tflops_median = kNaN;
  row.sm_clock_mhz_avg = kNaN;
  row.mem_clock_mhz_avg = kNaN;
  row.power_w_avg = kNaN;
  row.power_w_max = kNaN;
  row.temp_c_max = kNaN;
  row.clock_locked = options.clock_locked;
  row.clock_target_mhz = options.clock_target_mhz;
  row.correctness_checked = false;
  row.correctness_pass = false;
  row.check_mode = "not_checked";
  row.max_abs_error = kNaN;
  row.max_rel_error = kNaN;
  row.rel_l2_error = kNaN;
  row.gpu_name = gpu.name;
  row.gpu_uuid = gpu.uuid;
  row.sm_count = gpu.sm_count;
  row.compute_capability = std::to_string(gpu.major) + "." + std::to_string(gpu.minor);
  row.build_git_sha = options.build_git_sha;
  row.config_hash = options.config_hash;
  row.status = "NOT_RUN";
  return row;
}

void fill_correctness(RawRow& row, Correctness const& check) {
  row.correctness_checked = check.checked;
  row.correctness_pass = check.pass;
  row.check_mode = check.mode;
  row.max_abs_error = check.checked ? check.max_abs : kNaN;
  row.max_rel_error = check.checked ? check.max_rel : kNaN;
  row.rel_l2_error = check.checked ? check.rel_l2 : kNaN;
  row.mismatch_count = check.mismatch_count;
  row.nan_inf_count = check.nonfinite_count;
  row.checked_elements = check.checked_elements;
}

void fill_telemetry(RawRow& row, TelemetrySample const& sample) {
  if (!sample.valid) return;
  row.sm_clock_mhz_avg = sample.sm_clock_mhz;
  row.mem_clock_mhz_avg = sample.mem_clock_mhz;
  row.power_w_avg = sample.power_w;
  row.power_w_max = kNaN;  // endpoint average is not a maximum
  row.temp_c_max = sample.temperature_c;
}

std::string timer_scope_for(std::string const& effective_mode) {
  if (effective_mode == "wall") return "host_wall_including_launch";
  if (effective_mode == "graph") return "gpu_graph_replay_batch";
  return "gpu_kernel_batch";
}

int run_bench(Cli const& cli, GpuInfo const& gpu, cudaStream_t stream,
              TelemetryReader& telemetry, std::string const& started) {
  Options const& base = cli.options;
  std::vector<CaseDef> const all_cases = cli.case_defs.empty() ? default_cases() : cli.case_defs;
  std::map<std::string, CaseDef> by_id;
  for (auto const& c : all_cases) by_id[c.id] = c;

  std::vector<CaseDef> selected;
  std::vector<std::string> requested = cli.cases;
  if (requested.empty()) {
    if (base.mode == "full") {
      selected = all_cases;
    } else if (base.mode == "ncu") {
      selected.push_back(by_id.at("S01"));
    } else {
      selected.push_back(by_id.at("S01"));
      selected.push_back(by_id.at("S02"));
    }
  } else {
    for (auto const& id : requested) {
      auto it = by_id.find(id);
      if (it == by_id.end()) throw std::invalid_argument("unknown case id: " + id);
      selected.push_back(it->second);
    }
  }
  std::set<std::string> cold_ids(cli.cold_cases.begin(), cli.cold_cases.end());
  for (auto const& id : cold_ids) {
    if (!by_id.count(id)) throw std::invalid_argument("unknown cold case id: " + id);
  }

  std::vector<BackendFactory> factories;
  if (cli.impls.empty()) {
    factories = backend_registry();
  } else {
    for (auto const& name : cli.impls) {
      BackendFactory const* factory = find_backend(name);
      if (!factory) throw std::invalid_argument("unknown implementation: " + name);
      factories.push_back(*factory);
    }
  }
  for (auto const& factory : factories) {
    if (factory.layouts.empty()) continue;
    for (auto const& layout : cli.layouts) {
      if (std::find(factory.layouts.begin(), factory.layouts.end(), layout) ==
          factory.layouts.end()) {
        throw std::invalid_argument("implementation " + factory.name + " does not support " +
                                    layout);
      }
    }
  }

  std::vector<CaseOutcome> outcomes;
  outcomes.reserve(selected.size());
  int ok_rows = 0;
  int fail_rows = 0;
  int unsupported_rows = 0;
  int validated_cases = 0;
  int unchecked_rows = 0;

  for (CaseDef const& c : selected) {
    CaseOutcome outcome;
    outcome.def = c;
    std::printf("\n=== case %s (%dx%dx%d) ===\n", c.id.c_str(), c.m, c.n, c.k);
    std::fflush(stdout);

    Problem problem;
    problem.m = c.m;
    problem.n = c.n;
    problem.k = c.k;

    Buffer a(problem.a_elements() * sizeof(__half));
    Buffer b(problem.b_elements() * sizeof(__half));
    Buffer c_buffer(problem.c_elements() * sizeof(__half));
    DeviceBuffers buffers;
    buffers.a = a.value;
    buffers.b = b.value;
    buffers.c = c_buffer.value;
    buffers.a_bytes = a.bytes;
    buffers.b_bytes = b.bytes;
    buffers.c_bytes = c_buffer.bytes;

    HostOperands const host = make_host_operands(problem, base.seed);
    // 参考与输入都基于规范逻辑布局：A 为 M 连续，B 为 N 连续。
    Problem canonical = problem;
    canonical.ta = false;
    canonical.tb = true;
    std::printf("inputs generated; seed=%llu\n", (unsigned long long)base.seed);
    std::fflush(stdout);

    Reference reference;
    if (base.compute_reference) {
      reference = build_reference(canonical, host, base.device, stream, true);
      std::printf("reference=%s%s\n", reference.mode.c_str(),
                  reference.reason.empty() ? "" : (" (" + reference.reason + ")").c_str());
      std::fflush(stdout);
    }

    bool const need_flush = base.cache_mode == "cold" || cold_ids.count(c.id) > 0;
    Buffer flush_buffer(need_flush ? base.flush_bytes : 0);

    std::vector<std::string> cache_modes = {base.cache_mode};
    if (cold_ids.count(c.id) && base.cache_mode != "cold") cache_modes.push_back("cold");

    for (std::string const& cache_mode : cache_modes) {
      bool const cold = cache_mode == "cold";
      int const effective_inner = cold ? 1 : base.inner;
      std::string const effective_timer = timer_scope_for(base.timing_mode);

      for (std::string const& layout : cli.layouts) {
        Problem p = problem;
        p.ta = layout == "TN";
        p.tb = layout == "NT";
        std::printf("--- layout %s ---\n", layout.c_str());
        std::fflush(stdout);
        // 按目标物理布局重新打包并上传；计时区间不含这次拷贝。
        HostOperands const packed = pack_operands(p, host);
        upload_operands(buffers, packed, stream);

        struct Planned {
          BackendFactory const* factory = nullptr;
          std::unique_ptr<GemmPlan> plan;
          Correctness correctness;
          std::string status;
          std::string reason;
          Options options;
          std::vector<RoundMeasurement> rounds;
          bool timed = false;
          std::size_t outcome_index = 0;
        };
        std::vector<Planned> plans;
        plans.reserve(factories.size());

        for (BackendFactory const& factory : factories) {
          Planned item;
          item.factory = &factory;
          item.options = base;
          item.options.problem = p;
          item.options.cache_mode = cache_mode;
          std::size_t const oi = outcome.implementations.size();
          item.outcome_index = oi;
          outcome.implementations.push_back(ImplOutcome{factory.name, layout, cache_mode,
                                                        "NOT_RUN", "", 0, kNaN, kNaN});

          if (gpu.major < factory.min_cc_major ||
              (gpu.major == factory.min_cc_major && gpu.minor < factory.min_cc_minor)) {
            std::ostringstream reason;
            reason << "requires compute capability " << factory.min_cc_major << "."
                   << factory.min_cc_minor << ", device is " << gpu.major << "." << gpu.minor;
            item.status = "UNSUPPORTED";
            item.reason = reason.str();
            RawRow row = make_row(item.options, c, p, PlanInfo{factory.name, "", factory.accum},
                                  gpu, cache_mode, layout, effective_timer, 0, base.samples,
                                  effective_inner);
            row.status = "UNSUPPORTED";
            row.notes = reason.str();
            append_raw_row(base.csv_path, row);
            outcome.implementations[oi].status = "UNSUPPORTED";
            outcome.implementations[oi].reason = reason.str();
            outcome.implementations[oi].rows = 1;
            ++unsupported_rows;
            std::printf("  %-45s UNSUPPORTED (%s)\n", factory.name.c_str(), reason.str().c_str());
            std::fflush(stdout);
            continue;
          }

          Validator validator;
          if (reference.mode != "unavailable") {
            validator = [&]() {
              std::vector<float> output = download_half_as_float(
                  buffers.c, problem.c_elements(), stream);
              Correctness const check = validate_output(p, reference, output, c.limits(factory.accum));
              return check.checked && check.pass;
            };
          }

          try {
            if (factory.name.rfind("CUBLAS_LT", 0) == 0) {
              item.plan = make_cublas_lt(item.options, buffers, stream, factory.accum, validator);
            } else {
              item.plan = factory.make(item.options, buffers, stream, validator);
            }
          } catch (Skip const& error) {
            item.status = "UNSUPPORTED";
            item.reason = error.what();
            RawRow row = make_row(item.options, c, p,
                                  PlanInfo{factory.name, "", factory.accum}, gpu, cache_mode,
                                  layout, effective_timer, 0, base.samples, effective_inner);
            row.status = "UNSUPPORTED";
            row.notes = item.reason;
            append_raw_row(base.csv_path, row);
            outcome.implementations[oi].status = "UNSUPPORTED";
            outcome.implementations[oi].reason = item.reason;
            outcome.implementations[oi].rows = 1;
            ++unsupported_rows;
            std::printf("  %-45s UNSUPPORTED (%s)\n", factory.name.c_str(), item.reason.c_str());
            std::fflush(stdout);
            continue;
          } catch (std::exception const& error) {
            item.status = "FAIL";
            item.reason = error.what();
            RawRow row = make_row(item.options, c, p,
                                  PlanInfo{factory.name, "", factory.accum}, gpu, cache_mode,
                                  layout, effective_timer, 0, base.samples, effective_inner);
            row.status = "FAIL";
            row.notes = item.reason;
            append_raw_row(base.csv_path, row);
            outcome.implementations[oi].status = "FAIL";
            outcome.implementations[oi].reason = item.reason;
            outcome.implementations[oi].rows = 1;
            ++fail_rows;
            correctness_log(item.options, c, layout, factory.name, item.correctness, "FAIL",
                            item.reason);
            std::printf("  %-45s FAIL construct: %s\n", factory.name.c_str(),
                        item.reason.c_str());
            std::fflush(stdout);
            continue;
          }

          // 正确性先于性能。
          if (reference.mode != "unavailable") {
            initialize_output(buffers, stream);
            try {
              item.plan->run();
              CUDA_CHECK(cudaStreamSynchronize(stream));
              std::vector<float> output =
                  download_half_as_float(buffers.c, problem.c_elements(), stream);
              item.correctness = validate_output(p, reference, output, c.limits(factory.accum));
            } catch (std::exception const& error) {
              item.status = "FAIL";
              item.reason = std::string("correctness launch failed: ") + error.what();
            }
            if (item.status.empty() && !item.correctness.pass) {
              item.status = "FAIL";
              item.reason = item.correctness.reason.empty() ? "correctness failed"
                                                            : item.correctness.reason;
            }
            correctness_log(item.options, c, layout, item.plan->info.implementation,
                            item.correctness, item.status.empty() ? "PASS" : item.status,
                            item.reason);
          } else {
            item.correctness.mode = "unavailable";
            item.correctness.reason = reference.reason.empty() ? "reference disabled"
                                                               : reference.reason;
          }

          if (item.status == "FAIL") {
            RawRow row = make_row(item.options, c, p, item.plan->info, gpu, cache_mode, layout,
                                  effective_timer, 0, base.samples, effective_inner);
            fill_correctness(row, item.correctness);
            row.status = "FAIL";
            row.notes = item.reason;
            append_raw_row(base.csv_path, row);
            outcome.implementations[oi].status = "FAIL";
            outcome.implementations[oi].reason = item.reason;
            outcome.implementations[oi].rows = 1;
            ++fail_rows;
            std::printf("  %-45s FAIL correctness\n", factory.name.c_str());
            std::fflush(stdout);
            plans.push_back(std::move(item));
            continue;
          }

          std::printf("  %-45s correctness=%s\n", factory.name.c_str(),
                      item.correctness.checked ? "PASS" : "NOT_CHECKED");
          std::fflush(stdout);
          if (item.correctness.checked && item.correctness.pass) ++validated_cases;
          else if (base.skip_timing) ++unchecked_rows;
          outcome.implementations[oi].status = item.correctness.checked && item.correctness.pass
                                                   ? "OK" : "NOT_CHECKED";
          plans.push_back(std::move(item));
        }

        if (base.skip_timing) continue;

        for (int round = 0; round < base.rounds; ++round) {
          // 每轮轮换实现顺序。
          std::size_t const count = plans.size();
          for (std::size_t step = 0; step < count; ++step) {
            std::size_t const index = (step + std::size_t(round)) % count;
            Planned& item = plans[index];
            if (item.status == "FAIL" || !item.plan) continue;
            std::size_t const oi = item.outcome_index;

            Options timed = item.options;
            timed.rounds = 1;
            timed.samples = base.samples;
            timed.warmup = round == 0 ? base.warmup : 0;
            try {
              TimingResult const result =
                  benchmark_plan(*item.plan, p, timed, stream,
                                 flush_buffer.value ? &flush_buffer : nullptr, &telemetry);
              RoundMeasurement const& measurement = result.rounds.at(0);
              item.rounds.push_back(measurement);
              item.timed = true;

              RawRow row = make_row(item.options, c, p, item.plan->info, gpu, cache_mode, layout,
                                    timer_scope_for(result.effective_mode), round, base.samples,
                                    effective_inner);
              fill_correctness(row, item.correctness);
              fill_telemetry(row, measurement.telemetry);
              row.round_started_utc = measurement.started_utc;
              row.round_finished_utc = measurement.finished_utc;
              row.telemetry_scope = "round_endpoint_pair";
              row.duration_us_median = measurement.stats.median_us;
              row.duration_us_p10 = measurement.stats.p10_us;
              row.duration_us_p90 = measurement.stats.p90_us;
              row.duration_us_min = measurement.stats.min_us;
              row.duration_us_max = measurement.stats.max_us;
              row.tflops_median = measurement.tflops;
              row.status = item.correctness.checked && item.correctness.pass ? "OK" : "NOT_CHECKED";
              std::ostringstream notes;
              if (!item.plan->info.details.empty()) notes << item.plan->info.details << " | ";
              notes << result.note;
              if (!item.correctness.checked) notes << " | correctness=NOT_CHECKED";
              row.notes = notes.str();
              append_raw_row(base.csv_path, row);
              if (row.status == "OK") ++ok_rows;
              else ++unchecked_rows;
              outcome.implementations[oi].rows += 1;
              outcome.implementations[oi].duration_us_median = measurement.stats.median_us;
              outcome.implementations[oi].tflops_median = measurement.tflops;
              outcome.implementations[oi].status = row.status;
              std::printf("  round %d %-45s %9.3f us  %8.2f TFLOPS\n", round,
                          item.plan->info.implementation.c_str(), measurement.stats.median_us,
                          measurement.tflops);
            } catch (std::exception const& error) {
              RawRow row = make_row(item.options, c, p, item.plan->info, gpu, cache_mode, layout,
                                    effective_timer, round, base.samples, effective_inner);
              fill_correctness(row, item.correctness);
              row.status = "FAIL";
              row.notes = std::string("timing failed: ") + error.what();
              append_raw_row(base.csv_path, row);
              ++fail_rows;
              outcome.implementations[oi].status = "FAIL";
              outcome.implementations[oi].reason = row.notes;
              outcome.implementations[oi].rows += 1;
              std::printf("  round %d %-45s FAIL timing: %s\n", round,
                          item.plan->info.implementation.c_str(), error.what());
            }
            std::fflush(stdout);
          }
        }
      }
    }
    outcomes.push_back(std::move(outcome));
  }

  int const successful = base.skip_timing ? validated_cases : ok_rows;
  int const total = successful + fail_rows + unsupported_rows + unchecked_rows;
  std::string overall = "COMPLETED";
  if (total == 0) {
    overall = "FAILED";
  } else if ((fail_rows > 0 || unchecked_rows > 0) && successful == 0) {
    overall = "FAILED";
  } else if (fail_rows > 0 || unsupported_rows > 0 || unchecked_rows > 0) {
    overall = "PARTIAL";
  }
  std::printf("\nrows: ok=%d fail=%d unsupported=%d status=%s\n", ok_rows, fail_rows,
              unsupported_rows, overall.c_str());
  std::fflush(stdout);
  write_manifest(cli, gpu, started, overall, outcomes);
  return overall == "FAILED" ? 2 : 0;
}

void run_ncu_launch(Cli const& cli, GpuInfo const& gpu, cudaStream_t stream) {
  std::vector<CaseDef> all = cli.case_defs.empty() ? default_cases() : cli.case_defs;
  std::map<std::string, CaseDef> by_id;
  for (auto const& c : all) by_id[c.id] = c;
  std::string case_id = cli.cases.empty() ? "S01" : cli.cases.front();
  if (!by_id.count(case_id)) throw std::invalid_argument("unknown case id: " + case_id);
  CaseDef const& c = by_id.at(case_id);
  std::string layout = cli.layouts.empty() ? "TN" : cli.layouts.front();
  if (layout != "TN" && layout != "NT") throw std::invalid_argument("bad layout");
  if (cli.impls.size() != 1) throw std::invalid_argument("--stage ncu requires exactly one --impl");

  BackendFactory const* factory = find_backend(cli.impls.front());
  if (!factory) throw std::invalid_argument("unknown implementation: " + cli.impls.front());
  if (gpu.major < factory->min_cc_major) {
    std::printf("NCU_LAUNCH_SKIPPED case=%s impl=%s reason=compute_capability\n", case_id.c_str(),
                factory->name.c_str());
    return;
  }

  Problem p;
  p.m = c.m;
  p.n = c.n;
  p.k = c.k;
  p.ta = layout == "TN";
  p.tb = layout == "NT";

  Buffer a(p.a_elements() * sizeof(__half));
  Buffer b(p.b_elements() * sizeof(__half));
  Buffer out(p.c_elements() * sizeof(__half));
  DeviceBuffers buffers{a.value, b.value, out.value, a.bytes, b.bytes, out.bytes};
  HostOperands const logical = make_host_operands(p, cli.options.seed);
  HostOperands const packed = pack_operands(p, logical);
  upload_operands(buffers, packed, stream);
  initialize_output(buffers, stream);

  Options options = cli.options;
  options.problem = p;
  if (options.lt_algo_hex.empty()) options.lt_selection = "first";
  std::unique_ptr<GemmPlan> plan = factory->make(options, buffers, stream, Validator{});
  // 正式采集前先跑一次作为预热（NCU 用 --launch-skip 1 跳过它），
  // 避免把首次调用的懒加载/workspace 初始化计入 profile。
  plan->run();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int launch = 0; launch < 5; ++launch) plan->run();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::printf("NCU_LAUNCH_OK case=%s impl=%s layout=%s kernel=%s\n", case_id.c_str(),
              plan->info.implementation.c_str(), layout.c_str(),
              plan->info.kernel_name.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  Cli cli;
  try {
    cli = parse_cli(argc, argv);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "Argument error: %s\n", error.what());
    return 2;
  }

  if (cli.list_impls) {
    for (auto const& factory : backend_registry()) {
      std::printf("%-52s accum=%s layouts=TN,NT min_cc=%d.%d\n", factory.name.c_str(),
                  accum_name(factory.accum), factory.min_cc_major, factory.min_cc_minor);
    }
    return 0;
  }
  if (cli.list_cases) {
    for (auto const& c : (cli.case_defs.empty() ? default_cases() : cli.case_defs)) {
      std::printf("%s %s %d %d %d\n", c.id.c_str(), c.priority.c_str(), c.m, c.n, c.k);
    }
    return 0;
  }

  try {
    CUDA_CHECK(cudaSetDevice(cli.options.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cli.options.device));
    GpuInfo const gpu = gpu_info(prop);
    std::printf("Device: %s (%d SMs, compute %d.%d)\n", gpu.name.c_str(), gpu.sm_count,
                gpu.major, gpu.minor);
    std::printf("Run: plan=%s run_id=%s mode=%s stage=%s\n", cli.options.plan_id.c_str(),
                cli.options.run_id.c_str(), cli.options.mode.c_str(), cli.stage.c_str());
    std::fflush(stdout);

    Stream stream;
    TelemetryReader telemetry(cli.options.device);
    if (!telemetry.available()) {
      std::fprintf(stderr, "Warning: NVML telemetry unavailable: %s\n", telemetry.error().c_str());
    }
    std::string const started = now_utc();
    write_environment_json(cli, gpu, prop, telemetry, started);

    if (cli.stage == "ncu") {
      run_ncu_launch(cli, gpu, stream.value);
      return 0;
    }
    return run_bench(cli, gpu, stream.value, telemetry, started);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "Fatal: %s\n", error.what());
    return 2;
  }
}
