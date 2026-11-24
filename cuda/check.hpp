#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <cublas_v2.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace spark {

inline const char* cuda_message(CUresult status) {
    const char* message = "CUDA driver failure";
    if (cuGetErrorString(status, &message) != CUDA_SUCCESS || !message)
        return "CUDA driver failure";
    return message;
}
inline const char* cuda_message(cudaError_t status) { return cudaGetErrorString(status); }
inline const char* cuda_message(nvrtcResult status) { return nvrtcGetErrorString(status); }
inline const char* cuda_message(cublasStatus_t status) { return cublasGetStatusString(status); }

template <typename T>
void check_cuda(T status, const char* expression, const char* file, int line) {
    if (static_cast<int>(status) != 0)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " +
                                 expression + " failed: " + cuda_message(status));
}

template <typename T>
void cleanup_cuda(T status, const char* expression, const char* file, int line) noexcept {
    if (static_cast<int>(status) != 0)
        std::cerr << file << ':' << line << ' ' << expression << " failed: "
                  << cuda_message(status) << '\n';
}

}  // namespace spark

#define SPARK_CUDA(call) ::spark::check_cuda((call), #call, __FILE__, __LINE__)
#define SPARK_CUDA_CLEANUP(call) ::spark::cleanup_cuda((call), #call, __FILE__, __LINE__)
