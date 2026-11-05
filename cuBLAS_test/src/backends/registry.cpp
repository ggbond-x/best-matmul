#include "gemm/backend.hpp"

#include <algorithm>

namespace gemm {
namespace {

BackendFactory custom_entry(std::string name, std::string const& implementation,
                            std::function<std::unique_ptr<GemmPlan>(Options const&,
                                                                    DeviceBuffers const&,
                                                                    cudaStream_t)> make) {
  BackendFactory factory;
  factory.name = std::move(name);
  factory.accum = Accum::FP16;
  factory.layouts = {"TN", "NT"};
  factory.min_cc_major = 9;
  factory.min_cc_minor = 0;
  factory.make = [implementation, make](Options const& options, DeviceBuffers const& buffers,
                                        cudaStream_t stream, Validator const&) {
    (void)implementation;
    return make(options, buffers, stream);
  };
  return factory;
}

BackendFactory cublas_entry(std::string name, Accum accum, bool lt) {
  BackendFactory factory;
  factory.name = std::move(name);
  factory.accum = accum;
  factory.layouts = {"TN", "NT"};
  factory.make = [accum, lt](Options const& options, DeviceBuffers const& buffers,
                             cudaStream_t stream, Validator const& validator) {
    if (lt) return make_cublas_lt(options, buffers, stream, accum, validator);
    return make_cublas_ex(options, buffers, stream, accum);
  };
  return factory;
}

BackendFactory cutlass_entry(std::string name, Accum accum, std::string candidate) {
  BackendFactory factory;
  factory.name = std::move(name);
  factory.accum = accum;
  factory.layouts = {"TN", "NT"};
  factory.min_cc_major = 9;
  factory.min_cc_minor = 0;
  factory.make = [accum, candidate](Options const& options, DeviceBuffers const& buffers,
                                    cudaStream_t stream, Validator const&) {
    return make_cutlass_candidate(options, buffers, stream, accum, candidate);
  };
  return factory;
}

}  // namespace

std::vector<BackendFactory> const& backend_registry() {
  static std::vector<BackendFactory> const registry = [] {
    std::vector<BackendFactory> entries;
    entries.push_back(cublas_entry("CUBLAS_EX_F16ACC", Accum::FP16, false));
    entries.push_back(cublas_entry("CUBLAS_EX_F32ACC", Accum::FP32, false));
    entries.push_back(cublas_entry("CUBLAS_LT_F16ACC", Accum::FP16, true));
    entries.push_back(cublas_entry("CUBLAS_LT_F32ACC", Accum::FP32, true));

    entries.push_back(custom_entry("CUSTOM_V1_F16ACC", "v1", make_custom_v1));
    entries.push_back(custom_entry("CUSTOM_V2_PIPE2_F16ACC", "v2_pipe2", make_custom_v2_p2));
    entries.push_back(custom_entry("CUSTOM_V2_PIPE3_F16ACC", "v2_pipe3", make_custom_v2));
    entries.push_back(custom_entry("CUSTOM_V2_PIPE4_F16ACC", "v2_pipe4", make_custom_v2_p4));

    for (std::string const& candidate : {"128x128x64_C1x1x1", "128x256x64_C2x1x1"}) {
      entries.push_back(cutlass_entry("CUTLASS_SM90_" + candidate + "_F16ACC", Accum::FP16,
                                      candidate));
      entries.push_back(cutlass_entry("CUTLASS_SM90_" + candidate + "_F32ACC", Accum::FP32,
                                      candidate));
    }
    return entries;
  }();
  return registry;
}

BackendFactory const* find_backend(std::string const& name) {
  for (BackendFactory const& factory : backend_registry()) {
    if (factory.name == name) return &factory;
  }
  return nullptr;
}

}  // namespace gemm
