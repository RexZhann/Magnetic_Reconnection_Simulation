CXX      := g++
# -march=native：须在目标机器本地编译（cerberus3 与本机 CPU 不同，禁止拷贝二进制）
# -ffp-contract=off：禁 FMA 收缩，保证与旧二进制逐位一致（放开可再快一点但破坏位复现）
CXXFLAGS := -O3 -march=native -ffp-contract=off -std=c++17 -fopenmp -Iinclude -Wall -Wextra -pedantic -MMD -MP
LDFLAGS  := -fopenmp

SRC      := $(wildcard src/*.cpp)
OBJ      := $(patsubst src/%.cpp,build/%.o,$(SRC))
TEST_SRC := tests/test_main.cpp
TEST_BIN := build/test_main
APP_BIN  := build/mhd2d

.PHONY: all clean test run

all: $(APP_BIN)

build:
	mkdir -p build

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(APP_BIN): $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(TEST_BIN): $(TEST_SRC) $(filter-out build/main.o,$(OBJ)) | build
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

run: $(APP_BIN)
	./$(APP_BIN) 3 200 200 1 1

clean:
	rm -rf build *.dat

-include $(OBJ:.o=.d)
