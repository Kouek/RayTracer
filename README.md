# Ray Tracer

This project includes 2 parts: **Ray Tracer** and **Ray Caster**. All demos use GLFW + GLAD as the front-end display.

- Ray Tracer is an interactive CUDA-based Path Tracer, which implements the works
  - [Fast Parallel Construction of High-Quality Bounding Volume Hierarchies](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)
- Ray Caster is a CUDA-based renderer for accelerated Volume Rendering towards volumetric data with the format of RAW and VDB (GVDB), which implements the works
  - Part of [Fast Fluid Simulations with Sparse Volumes on the GPU](https://people.csail.mit.edu/kuiwu/gvdb_sim.html)
  - Depth-Box VDB

# Usage

## Build

- Clone this repository to the local.
- Make `3rd/glfw` a complete and valid repo from [GLFW](https://github.com/glfw/glfw)
- Run CMake configuration.
  - To show the performance, use a `Release` config.

## Input Arguments

### dvr_dpbx_vdb

This demo resamples the RAW volume into Depth-Box VDB or GVDB and performs Direct Volume Rendering on the volume.

#### Command Line Args

- Example: Head (from [GVDB](https://github.com/NVIDIA/gvdb-voxels
))

```json
[
  "-vt", "uint8",
  "-vol", "<Path to the volumetric data (RAW)>",
  "-dx", "128",
  "-dy", "256",
  "-dz", "256",
  "-tf", "<Path to the transfer function>",
  "-tx", "1",
  "-ty", "-3",
  "-tz", "-2",
  "-sx", "2.0",
  "-sy", "1.0",
  "-sz", "1.0"
]
```

- Example: Kingsnake (from [Open SciVis Datasets](https://klacansky.com/open-scivis-datasets/))

```json
[
    "-vt", "uint8",
    "-vol", "<Path to the volumetric data (RAW)>",
    "-dx", "1024",
    "-dy", "1024",
    "-dz", "795",
    "-tf", "<Path to the transfer function>"
]
```

#### Format of Transfer Function File

||scalar|r|g|b|a|
|--|--|--|--|--|--|
|range|[0, std::numeric_limits<VoxelType>::max()]|[0, 255]|[0, 255]|[0, 255]|[0, 255]|

- Example: Head

```txt
0.000000 0.000000 0.000000 0.000000 0.000000
49.840766 0.000000 0.000000 0.000000 0.000000
60.016020 5.041094 3.602943 52.500004 0.579299
127.955641 255.000000 255.000000 255.000000 5.000000
183.914396 8.000000 255.000000 0.000000 8.000000
255.000000 255.000000 0.000000 0.000000 10.000000
```

- Example: Kingsnake

```txt
0.000000 0.000000 0.000000 0.000000 0.000000
95.370346 0.000000 0.000000 0.000000 0.000000
125.946533 255.000000 255.000000 255.000000 20.000001
159.491972 255.000000 157.499998 0.000000 20.000001
232.292712 30.000000 0.000000 255.000000 100.000002
255.000000 255.000000 0.000000 0.000000 255.000000
```
