Backends
========

- cpu.py: NumPy-based reference backend (default)
- cuda.py: placeholder for optional CUDA backend (e.g., CuPy)

Code should prefer the runtime device API to select a backend.
