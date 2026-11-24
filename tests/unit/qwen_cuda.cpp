#include "cuda/qwen_gpu.hpp"
#include <exception>
#include <iostream>

int main() {
    try {
        spark::test_qwen_cuda_ops();
        std::cout << "CUDA Qwen operation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
