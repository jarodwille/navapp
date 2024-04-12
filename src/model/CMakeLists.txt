# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(model LANGUAGES CXX)

# to find libtorch
set(Torch_DIR "../../../libtorch/share/cmake/Torch")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-12.4/bin/nvcc")

# Find libtorch
find_package(Torch REQUIRED)

# Compile model.cc into a library (compiled during compilation time)
add_library(model_lib SHARED model.cc)
target_include_directories(model_lib PUBLIC .)
# target_compile_definitions(model_lib PUBLIC MODEL_LIB)

message (STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
# Link against libtorch 
target_link_libraries(model_lib ${TORCH_LIBRARIES})

install(TARGETS model_lib
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT shared NAMELINK_SKIP
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT development)

install(FILES model.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/valhalla"
  COMPONENT development)