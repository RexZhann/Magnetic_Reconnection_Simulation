# 2D MHD Project


## Build

```bash
make
```
or without make:
```bash
g++ -O3 -fopenmp -std=c++17 src/*.cpp -Iinclude -o mhd2d
```

## Run

```bash
./build/mhd2d <test> <nx> <ny> [glm] [solver]
```

Compatible with the original command line:
- `test`: 0=CylExplosion, 1=BrioWu-x, 2=BrioWu-y, 3=OrszagTang, 4=Rotor
- `glm`: 0=off, 1=on
- `solver`: 0=FORCE, 1=HLLD

Example:

```bash
OMP_NUM_THREADS=8 ./build/mhd2d 3 256 256 1 1
```

## Why this structure helps CT later

The current code uses a divergence controller interface. Right now it has:
- `NoDivBCleaning`
- `GLMDivergenceCleaning`

To add CT later, add a new class implementing `DivergenceController`, then move face-centered magnetic-field storage and EMF update into a dedicated CT module. The main time loop can stay mostly unchanged.
