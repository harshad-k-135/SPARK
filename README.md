# SPARK

SPARK (Specialized Public Accelerated Runtime for Kernels) targets deterministic
batch-one Llama inference in C++20 and CUDA C++ on the RTX 3050 6 GB Laptop GPU.
The initial model is Meta Llama 3.2 1B Instruct. SPARK's scope is one user and
one GPU. The implementation does its own forward pass.

See [results](docs/RESULTS.md) and the [benchmark protocol](docs/BENCHMARK_PROTOCOL.md).
