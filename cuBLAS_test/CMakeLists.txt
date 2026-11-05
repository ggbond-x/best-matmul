cmake_minimum_required(VERSION 3.16)
project(gemm_bench_plan1 LANGUAGES CXX CUDA)

# 统一构建：公共测试程序 + 库适配 + 自研 Kernel 适配。
# 计划参数（Shape、实现选择、计时）不在 CMake 中，放在 scripts/planN/plan.yaml。

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CMake 3.16 尚不支持 CMAKE_CUDA_STANDARD=17，直接用 nvcc 的 -std=c++17。
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

set(GEMM_CUDA_ARCH "90a" CACHE STRING "CUDA arch for the benchmark (plan1 target: 90a)")

# 兼容 CMake 3.16（无 CMAKE_CUDA_ARCHITECTURES）：直接生成单个 gencode，
# 避免 nvcc 同时编译默认架构导致 SM90 代码无法通过。
if(GEMM_CUDA_ARCH MATCHES "^(.*)a$")
  set(_gemm_compute_arch "compute_${CMAKE_MATCH_1}a")
  set(_gemm_code_arch "sm_${CMAKE_MATCH_1}a")
else()
  set(_gemm_compute_arch "compute_${GEMM_CUDA_ARCH}")
  set(_gemm_code_arch "sm_${GEMM_CUDA_ARCH}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=${_gemm_compute_arch},code=${_gemm_code_arch}")

# ---- CUTLASS -----------------------------------------------------------------
set(CUTLASS_ROOT "" CACHE PATH "CUTLASS repository root (contains include/cutlass)")
if(NOT CUTLASS_ROOT)
  if(DEFINED ENV{CUTLASS_ROOT} AND EXISTS "$ENV{CUTLASS_ROOT}/include/cutlass/cutlass.h")
    set(CUTLASS_ROOT "$ENV{CUTLASS_ROOT}")
  else()
    foreach(candidate
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../cutlass"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cutlass"
        "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass"
        "/usr/local/cutlass")
      if(EXISTS "${candidate}/include/cutlass/cutlass.h")
        set(CUTLASS_ROOT "${candidate}")
        break()
      endif()
    endforeach()
  endif()
endif()
if(NOT CUTLASS_ROOT OR NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "找不到 CUTLASS。请设置 -DCUTLASS_ROOT=/path/to/cutlass 或环境变量 CUTLASS_ROOT。")
endif()
message(STATUS "CUTLASS_ROOT=${CUTLASS_ROOT}")

# 与所选 nvcc 配套的 Toolkit 根目录（头文件/库/RPATH 都从这里取）。
get_filename_component(_gemm_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(_gemm_cuda_root "${_gemm_cuda_bin}" DIRECTORY)
set(_gemm_cuda_lib "${_gemm_cuda_root}/lib64")
message(STATUS "CUDA toolkit root=${_gemm_cuda_root}")

# ---- targets -----------------------------------------------------------------
set(GEMM_SOURCES
  src/main.cpp
  src/host_data.cpp
  src/timing.cpp
  src/csv.cpp
  src/telemetry.cpp
  src/backends/cublas_gemm.cpp
  src/backends/cublaslt_gemm.cpp
  src/backends/cutlass_gemm.cu
  src/backends/registry.cpp
  src/custom/custom_v1.cu
  src/custom/custom_v2_p2.cu
  src/custom/custom_v2_p3.cu
  src/custom/custom_v2_p4.cu
)

add_executable(gemm_bench ${GEMM_SOURCES})

target_include_directories(gemm_bench PRIVATE
  "${_gemm_cuda_root}/include"              # 与所选 nvcc 配套的 cuBLAS/cuBLASLt 头文件
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/.."          # kernels/ 与 host/ 的仓库相对路径
  "${CMAKE_CURRENT_SOURCE_DIR}/../kernels"  # 自研 Kernel 头文件
  "${CUTLASS_ROOT}/include"
  "${CUTLASS_ROOT}/tools/util/include"      # CUTLASS 4.x 的 cutlass/util/*.hpp
)

target_compile_options(gemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# Match the historical V1/V2 report configuration explicitly, not the shared
# kernel header's independent default (GMMA_INST_N=64).
target_compile_definitions(gemm_bench PRIVATE GMMA_INST_N=128)

# 自研 V2 的 PIPE 深度是编译期常量；每个 TU 只编译一种配置，保留原始 Kernel 源码。
set_source_files_properties(src/custom/custom_v2_p2.cu PROPERTIES
  COMPILE_DEFINITIONS "PIPE=2;WGMMA_MAX_INFLIGHT=2")
set_source_files_properties(src/custom/custom_v2_p3.cu PROPERTIES
  COMPILE_DEFINITIONS "PIPE=3;WGMMA_MAX_INFLIGHT=2")
set_source_files_properties(src/custom/custom_v2_p4.cu PROPERTIES
  COMPILE_DEFINITIONS "PIPE=4;WGMMA_MAX_INFLIGHT=2")

# cuBLAS/cuBLASLt 必须与使用的 nvcc 来自同一套 Toolkit；优先从编译器路径推导 lib64，
# 避免在同时装有旧版系统 CUDA 的机器上误链接到旧 cuBLAS。
find_library(CUBLAS_LIBRARY NAMES cublas HINTS "${_gemm_cuda_lib}" NO_DEFAULT_PATH)
find_library(CUBLASLT_LIBRARY NAMES cublasLt HINTS "${_gemm_cuda_lib}" NO_DEFAULT_PATH)
if(NOT CUBLAS_LIBRARY OR NOT CUBLASLT_LIBRARY)
  find_library(CUBLAS_LIBRARY NAMES cublas)
  find_library(CUBLASLT_LIBRARY NAMES cublasLt)
endif()
if(NOT CUBLAS_LIBRARY OR NOT CUBLASLT_LIBRARY)
  message(FATAL_ERROR "找不到 cuBLAS/cuBLASLt 库；请确认 CUDA Toolkit 安装完整。")
endif()
message(STATUS "CUBLAS_LIBRARY=${CUBLAS_LIBRARY}")
message(STATUS "CUBLASLT_LIBRARY=${CUBLASLT_LIBRARY}")
target_link_libraries(gemm_bench PRIVATE
  "${CUBLAS_LIBRARY}"
  "${CUBLASLT_LIBRARY}"
  ${CMAKE_DL_LIBS}
)
set_target_properties(gemm_bench PROPERTIES
  BUILD_RPATH "${_gemm_cuda_lib}"
  INSTALL_RPATH "${_gemm_cuda_lib}"
)

set_target_properties(gemm_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# 便于确认实际编译参数（写入 build/plan1/CMakeCache.txt 之外的单文件）。
file(WRITE "${CMAKE_BINARY_DIR}/cuda_arch.txt" "${GEMM_CUDA_ARCH}\n")
