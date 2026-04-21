#!/bin/bash
# Build bfs_expand binary using ParlayLib headers
PARLAY_INCLUDE=/home/ubuntu/xianglic/parlaylib/include
g++ -O3 -std=c++17 -I$PARLAY_INCLUDE \
    bfs_expand.cpp -o bfs_expand \
    -lpthread
echo "Built: bfs_expand"
