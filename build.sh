# will build to ./build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON_BINDINGS=ON
make -C build -j$(nproc)
sudo make -C build install