#include "gemm/telemetry.hpp"

#include <nvml.h>

#include <dlfcn.h>

#include <cstring>
#include <string>
#include <type_traits>

namespace gemm {
namespace {

// 运行时加载 NVML，避免在没有开发用 libnvidia-ml.so 符号链接的机器上链接失败。
// 加载失败只影响遥测字段，不影响基准本身。
struct NvmlApi {
  void* library = nullptr;
  decltype(&nvmlInit_v2) init = nullptr;
  decltype(&nvmlShutdown) shutdown = nullptr;
  decltype(&nvmlDeviceGetHandleByIndex_v2) handle_by_index = nullptr;
  decltype(&nvmlDeviceGetClockInfo) clock_info = nullptr;
  decltype(&nvmlDeviceGetPowerUsage) power_usage = nullptr;
  decltype(&nvmlDeviceGetTemperature) temperature = nullptr;
  decltype(&nvmlDeviceGetCurrentClocksThrottleReasons) throttle_reasons = nullptr;
  decltype(&nvmlErrorString) error_string = nullptr;
  std::string error;

  bool load() {
    for (char const* name : {"libnvidia-ml.so.1", "libnvidia-ml.so"}) {
      library = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (library) break;
    }
    if (!library) {
      char const* message = dlerror();
      error = std::string("dlopen NVML failed: ") + (message ? message : "unknown");
      return false;
    }
    auto resolve = [&](auto& slot, char const* name) {
      void* const symbol = dlsym(library, name);
      slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(symbol);
      if (!symbol) {
        error += std::string(error.empty() ? "" : "; ") + "missing NVML symbol " + name;
      }
      return symbol != nullptr;
    };
    bool ok = true;
    ok &= resolve(init, "nvmlInit_v2");
    ok &= resolve(shutdown, "nvmlShutdown");
    ok &= resolve(handle_by_index, "nvmlDeviceGetHandleByIndex_v2");
    ok &= resolve(clock_info, "nvmlDeviceGetClockInfo");
    ok &= resolve(power_usage, "nvmlDeviceGetPowerUsage");
    ok &= resolve(temperature, "nvmlDeviceGetTemperature");
    ok &= resolve(throttle_reasons, "nvmlDeviceGetCurrentClocksThrottleReasons");
    ok &= resolve(error_string, "nvmlErrorString");
    return ok;
  }
};

NvmlApi& nvml_api() {
  static NvmlApi api;
  static bool const loaded = api.load();
  (void)loaded;
  return api;
}

std::string nvml_message(NvmlApi const& api, nvmlReturn_t status) {
  if (api.error_string) {
    char const* text = api.error_string(status);
    if (text) return text;
  }
  return "status " + std::to_string(int(status));
}

}  // namespace

TelemetryReader::TelemetryReader(int device_index) {
  NvmlApi& api = nvml_api();
  if (!api.library || !api.init) {
    error_ = api.error.empty() ? "NVML library unavailable" : api.error;
    return;
  }
  nvmlReturn_t status = api.init();
  if (status != NVML_SUCCESS) {
    error_ = "nvmlInit: " + nvml_message(api, status);
    return;
  }
  nvmlDevice_t device = nullptr;
  status = api.handle_by_index(unsigned(device_index), &device);
  if (status != NVML_SUCCESS) {
    error_ = "nvmlDeviceGetHandleByIndex: " + nvml_message(api, status);
    api.shutdown();
    return;
  }
  device_ = device;
  available_ = true;
}

TelemetryReader::~TelemetryReader() {
  if (available_) nvml_api().shutdown();
}

TelemetrySample TelemetryReader::sample() const {
  TelemetrySample result;
  NvmlApi const& api = nvml_api();
  if (!available_ || !api.library) {
    result.reason = error_;
    return result;
  }
  nvmlDevice_t device = static_cast<nvmlDevice_t>(device_);
  unsigned clock = 0;
  bool any = false;
  if (api.clock_info(device, NVML_CLOCK_SM, &clock) == NVML_SUCCESS) {
    result.sm_clock_mhz = double(clock);
    any = true;
  }
  if (api.clock_info(device, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS) {
    result.mem_clock_mhz = double(clock);
    any = true;
  }
  unsigned power_mw = 0;
  if (api.power_usage(device, &power_mw) == NVML_SUCCESS) {
    result.power_w = double(power_mw) / 1000.0;
    any = true;
  }
  unsigned temperature = 0;
  if (api.temperature(device, NVML_TEMPERATURE_GPU, &temperature) == NVML_SUCCESS) {
    result.temperature_c = double(temperature);
    any = true;
  }
  unsigned long long reasons = 0;
  if (api.throttle_reasons(device, &reasons) == NVML_SUCCESS) {
    result.throttle_reasons = reasons;
    any = true;
  }
  result.valid = any;
  if (!any) result.reason = "NVML sampling returned no fields";
  return result;
}

}  // namespace gemm
