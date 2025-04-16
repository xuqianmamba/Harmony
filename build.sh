#!/bin/bash
cmake -B release -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/es01/home/lvxg/vdb/gcc-12.1.0/bin/gcc -DCMAKE_CXX_COMPILER=/es01/home/lvxg/vdb/gcc-12.1.0/bin/g++ . && cmake --build release -j
