#pragma once
// NVML 逐轮采样：SM/显存时钟、功耗、温度、节流原因。
// 采样失败不影响基准；CSV 中对应字段写 N/A。
#include <cstdint>
#include <limits>
#include <string>

namespace gemm {

struct TelemetrySample {
  bool valid = false;
  double sm_clock_mhz = std::numeric_limits<double>::quiet_NaN();
  double mem_clock_mhz = std::numeric_limits<double>::quiet_NaN();
  double power_w = std::numeric_limits<double>::quiet_NaN();
  double temperature_c = std::numeric_limits<double>::quiet_NaN();
  unsigned long long throttle_reasons = 0;
  std::string reason;
};

class TelemetryReader {
 public:
  explicit TelemetryReader(int device_index);
  ~TelemetryReader();
  TelemetryReader(TelemetryReader const&) = delete;
  TelemetryReader& operator=(TelemetryReader const&) = delete;

  bool available() const { return available_; }
  std::string const& error() const { return error_; }
  TelemetrySample sample() const;

 private:
  bool available_ = false;
  std::string error_;
  void* device_ = nullptr;  // nvmlDevice_t，避免在公共头文件中引入 nvml.h
};

}  // namespace gemm
