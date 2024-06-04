#!/bin/bash
set -e

# Clean.
rm -rf ./build

# Download ignition transcripts.
cd ./srs_db
./download_ignition.sh 3
cd ..

# Pick native toolchain file.
TOOLCHAIN=x86_64-linux-gcc10

# Build native.
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithAssert -DTOOLCHAIN=$TOOLCHAIN -DCMAKE_C_COMPILER=/usr/bin/gcc-10 -DCMAKE_CXX_COMPILER=/usr/bin/gcc-10 ..
cmake --build . --parallel
cd ..
