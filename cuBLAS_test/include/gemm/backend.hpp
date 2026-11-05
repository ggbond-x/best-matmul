#pragma once
// 后端统一接口：构造时完成句柄/描述符/算法搜索，run() 只提交一次 GEMM。
// 所有实现共享同一组 A/B/C 设备缓冲区，因此对比使用完全相同的输入。
#include "gemm/config.hpp"
#include "gemm/cuda_utils.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gemm {

class GemmPlan {
 public:
  virtual ~GemmPlan() = default;
  virtual void run() = 0;
  PlanInfo info;
};

// cuBLASLt 候选算法选择时的正确性回调：检查当前 C 缓冲区是否通过参考比较。
using Validator = std::function<bool()>;

struct BackendFactory {
  std::string name;
  Accum accum = Accum::FP32;
  std::vector<std::string> layouts;  // 支持 "TN"/"NT"；空表示两者都支持
  int min_cc_major = 0;
  int min_cc_minor = 0;
  std::function<std::unique_ptr<GemmPlan>(Options const&, DeviceBuffers const&,
                                          cudaStream_t, Validator const&)> make;
};

std::vector<BackendFactory> const& backend_registry();
BackendFactory const* find_backend(std::string const& name);

// 以下工厂分布在 src/backends 与 src/custom 中。
std::unique_ptr<GemmPlan> make_cublas_ex(Options const&, DeviceBuffers const&, cudaStream_t, Accum);
std::unique_ptr<GemmPlan> make_cublas_lt(Options const&, DeviceBuffers const&, cudaStream_t,
                                         Accum, Validator const&);

std::unique_ptr<GemmPlan> make_custom_v1(Options const&, DeviceBuffers const&, cudaStream_t);
std::unique_ptr<GemmPlan> make_custom_v2(Options const&, DeviceBuffers const&, cudaStream_t);
std::unique_ptr<GemmPlan> make_custom_v2_p2(Options const&, DeviceBuffers const&, cudaStream_t);
std::unique_ptr<GemmPlan> make_custom_v2_p4(Options const&, DeviceBuffers const&, cudaStream_t);

// candidate 取值：T128x128x64_C1x1x1、T128x256x64_C2x1x1
std::unique_ptr<GemmPlan> make_cutlass_candidate(Options const&, DeviceBuffers const&,
                                                 cudaStream_t, Accum, std::string const& candidate);

}  // namespace gemm
