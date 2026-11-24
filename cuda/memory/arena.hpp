#pragma once

#include <cuda.h>
#include <cstddef>

namespace spark {

class DeviceArena {
public:
    explicit DeviceArena(std::size_t bytes);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    CUdeviceptr allocate(std::size_t bytes);
private:
    CUdeviceptr base_ = 0;
    std::size_t bytes_ = 0, used_ = 0;
};

}  // namespace spark
