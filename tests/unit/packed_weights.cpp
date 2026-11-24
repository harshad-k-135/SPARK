#include "engine/model_loader/packed_weights.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

void expect(float value, float expected) {
    if (value != expected) {
        throw std::runtime_error("decoded " + std::to_string(value) +
                                 " instead of " + std::to_string(expected));
    }
}

void test_k45(bool high_bits) {
    std::array<std::uint8_t, 176> input{};
    std::array<float, 256> output{};
    input[1] = 0x3c;  // d = 1
    input[3] = 0x38;  // dmin = 0.5
    input[4] = 0x82;  // group 0 scale 2, group 4 scale high bits 2
    input[8] = 0x41;  // group 0 min 1, group 4 min high bits 1
    input[12] = 0x43; // group 4 scale low bits 3, min low bits 4
    input[high_bits ? 48 : 16] = 0x05;
    input[high_bits ? 112 : 80] = 0x07;
    if (high_bits) {
        input[16] = (1U << 4) | 1U;
    }
    spark::unpack_gguf_row(high_bits ? 13 : 12,
                           std::span(input).first(high_bits ? 176 : 144), output);
    expect(output[0], high_bits ? 41.5F : 9.5F);
    expect(output[128], high_bits ? 795.0F : 235.0F);
}

void test_k6() {
    std::array<std::uint8_t, 210> input{};
    std::array<float, 256> output{};
    input[0] = 0xC1;
    input[128] = 0x12;
    input[192] = 2;
    input[196] = 0xFD;
    input[209] = 0x38;
    spark::unpack_gguf_row(14, input, output);
    expect(output[0], 1.0F);
    expect(output[64], 6.0F);
}

void test_q8() {
    std::array<std::uint8_t, 34> input{};
    std::array<float, 32> output{};
    input[1] = 0x38;
    input[2] = 0x80;
    input[33] = 0x7f;
    spark::unpack_gguf_row(8, input, output);
    expect(output[0], -64.0F);
    expect(output[31], 63.5F);
    try {
        spark::unpack_gguf_row(8, std::span(input).first(33), output);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("truncated Q8_0 row was accepted");
}

void test_scalars() {
    std::array<float, 2> output{};
    spark::unpack_gguf_row(0, std::array<std::uint8_t, 8>{0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0}, output);
    expect(output[0], 1.0F);
    expect(output[1], -2.0F);
    spark::unpack_gguf_row(1, std::array<std::uint8_t, 4>{1, 0, 0, 0xc0}, output);
    expect(output[0], 0x1p-24F);
    expect(output[1], -2.0F);
    spark::unpack_gguf_row(30, std::array<std::uint8_t, 4>{0x80, 0x3f, 0, 0xc0}, output);
    expect(output[0], 1.0F);
    expect(output[1], -2.0F);
}

}  // namespace

int main() {
    try {
        test_k45(false);
        test_k45(true);
        test_k6();
        test_q8();
        test_scalars();
        std::cout << "Packed weight decoder tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
