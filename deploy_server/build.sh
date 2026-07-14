#!/usr/bin/env bash
# cerberus3 build: bare g++ -O3 -fopenmp -march=native (no scheduler, no modules)
[ "$(hostname)" = "cerberus3" ] || { echo "WRONG HOST: $(hostname)"; exit 1; }
echo "host=$(hostname) nproc=$(nproc) load=$(uptime)" | tee -a run.log
df -h $HOME | tee -a run.log && quota -s | tee -a run.log
set -euo pipefail
cd "$(dirname "$0")/.."
make -j"$(nproc)" CXXFLAGS="-O3 -std=c++17 -fopenmp -Iinclude -march=native -Wall"
echo "build OK -> build/mhd2d"
./build/mhd2d 2>/dev/null | head -1 || true
