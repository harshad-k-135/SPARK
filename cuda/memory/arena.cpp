#include "cuda/memory/arena.hpp"
#include "cuda/check.hpp"

#include <stdexcept>

namespace spark {

DeviceArena::DeviceArena(std::size_t bytes) : bytes_(bytes) {
    std::size_t available = 0, total = 0;
    SPARK_CUDA(cudaMemGetInfo(&available, &total));
    constexpr std::size_t reserve = 512ULL * 1024 * 1024;
    if (available <= reserve || bytes > available - reserve)
        throw std::runtime_error("CUDA memory budget exceeded. Required " +
            std::to_string(bytes) + " bytes plus 512 MiB headroom, " +
            std::to_string(available) + " bytes available");
    SPARK_CUDA(cuMemAlloc(&base_, bytes));
}

DeviceArena::~DeviceArena() {
    if (base_) SPARK_CUDA_CLEANUP(cuMemFree(base_));
}

CUdeviceptr DeviceArena::allocate(std::size_t bytes) {
    constexpr std::size_t alignment = 256;
    auto start = (used_ + alignment - 1) & ~(alignment - 1);
    if (start > bytes_ || bytes > bytes_ - start)
        throw std::runtime_error("device arena exhausted");
    used_ = start + bytes;
    return base_ + start;
}

}  // namespace spark
