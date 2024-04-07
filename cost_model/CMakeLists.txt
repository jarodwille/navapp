# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(train_model LANGUAGES CXX)

# to find libtorch
set(Torch_DIR "../../libtorch/share/cmake/Torch")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-12.4/bin/nvcc")

# Find libtorch
find_package(Torch REQUIRED)

# Add train_model executable
add_executable(train_model train_model.cc)

# Link against libtorch
target_link_libraries(train_model ${TORCH_LIBRARIES})

