Habitat C++ Sources
===================
This directory contains the C++ source code for Habitat. All the C++ code is
kept in this unified directory to simplify code sharing. CMake is used to
compile the code.

The C++ code is currently used to build two targets:

- `habitat_cuda`: A Python-importable module that provides bindings to the
  CUDA-related functionality used by Habitat.
- `device_info`: A utility that prints information about the underlying GPU
  (e.g., number of SMs, memory bandwidth, etc.).

Code Organization
-----------------
Each target corresponds to one file under the `src` directory. The rest of the
supporting code is organized in subdirectories under `src`.

If code is shared among multiple targets, it will likely be organized into a
independently compiled library that can be linked to the needed targets.
