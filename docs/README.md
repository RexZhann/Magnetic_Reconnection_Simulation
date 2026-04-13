# 2D MHD Project


## Build

```bash
make
```
or without make:
```bash
g++ -O3 -fopenmp -std=c++17 src/*.cpp -Iinclude -o mhd2d
```

Windows PowerShell:

```powershell
C:\TDM-GCC-64\bin\g++.exe -O3 -std=c++17 -fopenmp -Iinclude src\divergence_control.cpp src\main.cpp src\riemann.cpp src\solver.cpp src\state.cpp -o build\mhd2d.exe
```

Build tests on Windows PowerShell:

```powershell
C:\TDM-GCC-64\bin\g++.exe -O3 -std=c++17 -fopenmp -Iinclude tests\test_main.cpp src\divergence_control.cpp src\riemann.cpp src\solver.cpp src\state.cpp -o build\test_main.exe
```

## Run

```bash
./build/mhd2d <test> <nx> <ny> [glm] [solver]
```

Compatible with the original command line:
- `test`: 0=CylExplosion, 1=BrioWu-x, 2=BrioWu-y, 3=OrszagTang, 4=Rotor
- `glm`: 0=None, 1=GLM, 2=CT
- `solver`: 0=FORCE, 1=HLLD

Example:

```bash
OMP_NUM_THREADS=8 ./build/mhd2d 3 256 256 1 1
```

Windows PowerShell example:

```powershell
build\mhd2d.exe 0 32 32 2 1
```
