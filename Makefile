CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Iinclude -Wall -Wextra -pedantic
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
