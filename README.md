# Ray Tracer

This project includes 2 parts: **Ray Tracer** and **Ray Caster**. All demos use GLFW + GLAD as the front-end display.

- Ray Tracer is an interactive CUDA-based Path Tracer, which implements the works
  - [Fast Parallel Construction of High-Quality Bounding Volume Hierarchies](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)
- Ray Caster is a CUDA-based renderer for accelerated Volume Rendering towards volumetric data with the format of RAW and VDB (GVDB), which implements the works
  - Part of [Fast Fluid Simulations with Sparse Volumes on the GPU](https://people.csail.mit.edu/kuiwu/gvdb_sim.html)
  - Depth-Box VDB
