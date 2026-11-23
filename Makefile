# 2D MHD reconnection code -- single build entry point.
#
#   make                 build build/mhd2d
#   make test            build + run unit tests
#   make run             build + run with config/ctrl.txt (CTRL=... to change)
#   make clean           remove build/
#   make help            this list
#
# Knobs (override on the command line, e.g. `make ARCH= DEBUG=1`):
#   CXX    compiler (default g++; on macOS Apple clang + Homebrew libomp is auto-detected)
#   ARCH   CPU tuning flag (default -march=native: compile on the machine that runs it;
#          binaries must not be copied between laptop and cluster)
#   DEBUG  1 = -O0 -g with sanitizers off
#   OMP_PREFIX  libomp install prefix for Apple clang (default: brew --prefix libomp)

CXX      ?= g++
ARCH     ?= -march=native
DEBUG    ?= 0
CTRL     ?= config/ctrl.txt

# -ffp-contract=off: no FMA contraction -> bitwise reproducible against older runs
OPTFLAGS := -O3 $(ARCH) -ffp-contract=off
ifeq ($(DEBUG),1)
  OPTFLAGS := -O0 -g
endif

CXXFLAGS += $(OPTFLAGS) -std=c++17 -Iinclude -Wall -Wextra -pedantic -MMD -MP
LDFLAGS  +=

# --- OpenMP ------------------------------------------------------------------
# GCC: -fopenmp. Apple clang has no bundled OpenMP: use Homebrew's libomp
# (brew install libomp), or build with a real GCC (make CXX=g++-15).
IS_APPLE_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -q 'Apple clang' && echo 1)
ifeq ($(IS_APPLE_CLANG),1)
  OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
  ifeq ($(wildcard $(OMP_PREFIX)/include/omp.h),)
    OMP_MISSING := 1
  endif
  CXXFLAGS += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  LDFLAGS  += -L$(OMP_PREFIX)/lib -lomp -Wl,-rpath,$(OMP_PREFIX)/lib
else
  CXXFLAGS += -fopenmp
  LDFLAGS  += -fopenmp
endif

# --- sources -------------------------------------------------------------------
SRC      := $(wildcard src/*.cpp)
OBJ      := $(patsubst src/%.cpp,build/%.o,$(SRC))
APP_BIN  := build/mhd2d
TEST_BIN := build/test_main

.PHONY: all test run clean help check-omp

all: $(APP_BIN)

check-omp:
ifeq ($(OMP_MISSING),1)
	$(error Apple clang needs libomp: run `brew install libomp`, or pass OMP_PREFIX=/path, or CXX=g++-15)
endif

build:
	mkdir -p build

build/%.o: src/%.cpp | build check-omp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(APP_BIN): $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(TEST_BIN): tests/test_main.cpp $(filter-out build/main.o,$(OBJ)) | build
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

run: $(APP_BIN)
	./$(APP_BIN) $(CTRL)

clean:
	rm -rf build

help:
	@sed -n '3,15p' Makefile

-include $(OBJ:.o=.d) build/test_main.d
