# Changelog

All notable changes to forgeNN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (Docs) Clarified improved `mse` behavior (auto one-hot + regression reshaping) and availability of `model.summary()` in README.
- Modular optimizer system with new classes: `SGD`, `Adam`, and `AdamW` (decoupled weight decay)
- Deferred optimizer parameter binding (`opt = Adam(lr=1e-3)` then pass instance directly to `compile`)
- Ability to pass optimizer instances OR dict configs (`{"type":"adam", "lr":1e-3}` still supported)
- Optimizer public base class `Optimizer` exported at top level
- New example `optimizer_convergence_demo.py` comparing SGD vs Adam convergence
- Unit tests planned for optimizer correctness (bias correction, decoupled decay) added in test suite

### Changed
- Optimized training loop: removed redundant full-dataset forward after each epoch; metrics now aggregated on-the-fly (no API change).
- Fused stable softmax + cross-entropy implementation reduces intermediate allocations and duplicate exponentials.
- Documentation and guides updated to reflect new optimizer API (Sequential & Training guides, Comparison guide)

### Fixed
- Ensured `mse` consistently handles (N,) integer class targets vs. (N,C) logits without user-side one-hot conversion.

### Deprecated
- Legacy `VectorizedOptimizer` name now an alias of `SGD` (will be removed in a future major release)

### Performance
- Minor speed improvement from fused cross-entropy and eliminated extra per-epoch evaluation pass.

### Notes
- No breaking changes; public APIs (`compile().fit`, losses, metrics) unchanged.
- Dropped experimental acceleration path (Numba) before release—kept code lean and dependency surface minimal.

## [1.2.1] - 2025-09-09

### Added
- Keras-like `Sequential.summary()` method with symbolic shape inference:
  - Displays layer type (including attached activation), inferred output shape, and parameter counts
  - Automatically initializes lazily defined `Dense` layers when input feature size can be inferred
  - Supports optional `Input` layer to seed shape propagation or an explicit `input_shape` argument
- New `Input` layer (shape placeholder) exported at top level; integrates with summary and symbolic inference
- Auto one-hot + regression reshaping logic in `mse` improved (classification targets converted when 1D indices and logits are 2D)

### Changed
- Restored lazy initialization semantics for `Dense` during forward, while allowing summary to perform safe symbolic initialization when shapes are fully known
- Summary now keeps unresolvable shapes as `?` instead of forcing initialization (safer for partially dynamic pipelines)
- Documentation (guides) updated to reflect `Input` layer usage and model introspection via `model.summary()`

### Fixed
- Resolved missing `Sequential.summary` attribute caused by earlier nested definition bug
- Ensured parameter counts include activation-wrapped layers consistently

### Notes
- No breaking API changes; existing models continue to work
- `Input` layer is optional—models without it still summarize if `input_shape` is passed or shapes become inferable after first forward

### Upcoming (Planned)
- Potential inclusion of concatenation (`cat`) and stacking utilities per TODO roadmap
- Extended summary statistics (dtype, trainable flags) in future minor release


## [1.2.0] - 2025-09-08

### Added
- High-level layering API:
  - `Layer` base class and `ActivationWrapper` using the `@` operator to attach activations
  - `Sequential` container for ordered composition of layers
  - `Dense` (lazy init) and `Flatten` layers
  - Placeholders for `Conv2D` and `MaxPool2D` (raise NotImplementedError for now)
- Keras-like trainer utilities:
  - `compile(model, optimizer, loss, metrics)` returning a `CompiledModel`
  - `fit`, `evaluate`, and `predict` helpers
  - Built-in loss registry (`cross_entropy`) and metric registry (`accuracy`)
  - Mean Squared Error loss (`mse`) implemented in `forgeNN.vectorized` and registered in the trainer loss registry (usable via `loss="mse"` in `compile`)
- Documentation & guides:
  - New `SEQUENTIAL_GUIDE.md` explaining the Sequential API with comparisons to PyTorch and Keras
  - New `TRAINING_GUIDE.md` mapping Keras/PyTorch training to `compile/fit`
  - Updated `README.md` with `compile/fit` examples and guide links
- Examples:
  - New `examples/sequential_mnist.py` using `Sequential` and `compile/fit`
  - Updated `examples/benchmark_mnist.py` to use `Sequential + compile/fit` while keeping PyTorch comparison and plots
- Testing:
  - Expanded unit tests for `Sequential`, `Dense` lazy init, activation handling (string/class/callable), `Flatten`, nested `Sequential.zero_grad`, and optimizer momentum buffers
  - Ensured optimizer works with lazy-initialized `Dense` by performing a dummy forward prior to optimizer construction
- Public API:
  - Exported `Sequential`, `Dense`, `Flatten`, `ActivationWrapper`, and `compile` from the top-level package

### Changed
- Trainer stability and metric aggregation:
  - `CompiledModel.fit`/`evaluate` now aggregate loss and metrics weighted by batch size
  - `evaluate` computes accuracy via total correct/total samples for exact dataset accuracy
- Benchmark updates:
  - `examples/benchmark_mnist.py` now seeds NumPy for reproducibility and uses batch size 64
  - Per-epoch metrics are collected via `evaluate` to reduce jitter in plots
- Documentation/docstrings:
  - Comprehensive docstrings across `tensor.py` and `vectorized.py` with cleaned examples
  - Fixed `cross_entropy_loss` docstring to show correct gradient shape `(2, 2)`
 - Demo notebook training settings tuned for small datasets (increased epochs, reduced batch size, slightly higher learning rate) and updated to use the current `evaluate(X, y)` signature

### Deprecated
- Direct use of `VectorizedMLP`+manual training loops in examples in favor of `Sequential + compile/fit`
  - These APIs remain available for 1.x but are slated for removal/refactor in a future 2.x release
- Legacy training snippets in guides superseded by `compile/fit` sections

### Fixed
- Accuracy wobble in benchmarks by switching to sample-weighted aggregation and exact accuracy counting
- Lazy-init + optimizer ordering issue in tests by adding a dummy forward before optimizer/trainer creation
 - Implemented `mse` using graph composition (`(pred - target)**2 .mean()`) to ensure correct gradient scaling across all elements and robust broadcasting
 - Prevented unintended broadcasting for single-logit outputs by reshaping 1D targets `(N,)` to `(N, 1, ...)` when logits have singleton non-batch dimensions

### Security
- No security-related changes in this release

## [1.1.1] - 2025-09-08
### Added
- Added comprehensive documentation for all of the methods. Now the API is fully documented. If you see any missing docstrings, please open an issue.
- Added `elu()` activation function with proper tensor integration
- Added unit tests for `elu()` function

### Fixed
- 

## [1.1.0.1] - 2025-09-08
### Added
- Added `dot()` method for 1D tensor dot product with autograd support
- Added `GELU` activation function with proper tensor integration
- Added unit tests for `dot()` and `GELU` functions

### Fixed
- Fixed minor bug in `VectorizedLayer` activation handling
- Improved error messages for tensor shape operations
- Enhanced documentation for new tensor methods

## [1.1.0] - 2025-09-07

### MAJOR CLEANUP - Legacy Removal
- **REMOVED**: Entire `forgeNN.legacy` module - no longer maintained
- **REMOVED**: Backward compatibility code and comments
- **CLEAN**: Simplified API focused on modern, high-performance components
- **FOCUS**: Pure vectorized operations for maximum speed

### Enhanced
- **Unified activation system**: String/class/callable activations fully integrated
- **Cleaner documentation**: Removed outdated legacy references
- **Modern API**: Streamlined imports and cleaner codebase

## [1.0.4a0] - 2025-09-07

### Added
- **Activation function integration**: Full tensor integration for all activation functions
- **New tensor methods**: `leaky_relu()` and `swish()` with proper gradients
- **Enhanced VectorizedLayer**: Supports string, class, and callable activations
- **Clean loss API**: Removed confusing `functions.loss` module

## [1.0.4] - 2025-09-07

### Added
- **Complete tensor shape operations suite**
  - `reshape()` - Change tensor dimensions with automatic size inference
  - `view()` - Memory-efficient reshape operation
  - `flatten()` - Convert multidimensional tensors to 1D
  - `contiguous()` - Ensure memory contiguity for operations
  - `transpose()` - Swap tensor dimensions with proper gradient flow
  - `squeeze()` - Remove dimensions of size 1
  - `unsqueeze()` - Add dimensions of size 1
- **Matrix multiplication operations**
  - Full `@` operator support with broadcasting
  - Proper gradient computation for backpropagation
  - Support for batch matrix operations
- **Comprehensive test suite**
  - 40 unit tests covering all tensor operations
  - Complete gradient flow validation
  - Edge case testing and error handling
  - 100% test pass rate
- **Performance benchmarks and documentation**
  - MNIST benchmark showing 2.10x speedup over PyTorch
  - Detailed PyTorch vs forgeNN comparison guide
  - Live demo script for framework comparison
  - Professional documentation and examples

### Enhanced
- **VectorizedMLP performance improvements**
  - Optimized for small models (3.52x faster than PyTorch on <109k parameters)
  - Better accuracy on MNIST (97.37% vs PyTorch's 97.30%)
  - Xavier weight initialization for improved training dynamics
- **Gradient computation reliability**
  - Fixed dimension tracking in squeeze operations
  - Improved numerical stability in loss functions
  - Enhanced autograd system for complex operation chains
- **Code quality and organization**
  - Clean repository structure with professional .gitignore
  - Comprehensive error handling and validation
  - Consistent API design across all operations

### Documentation
- **README.md** - Updated with performance highlights and quick start guide
- **COMPARISON_GUIDE.md** - Detailed framework comparison with benchmarks
- **Comprehensive examples** - Complete MNIST classification example
- **API documentation** - Clear docstrings for all public methods

### Technical Improvements
- All tensor operations now support proper autograd
- Memory-efficient implementations for large tensor operations
- Robust error handling for shape mismatches
- Professional logging and debugging support

### Performance
- 2.10x faster MNIST training compared to PyTorch
- 3.52x speedup on small models (<109k parameters)
- Efficient vectorized operations using NumPy backend
- Minimal memory overhead for gradient computation

### Breaking Changes
None - All changes are backward compatible

### Migration Guide
No migration required - existing code will continue to work unchanged.

## [1.0.0] - 2025-09-06

### Added
- Initial release of forgeNN framework
- Vectorized automatic differentiation with `Tensor` class
- High-performance `VectorizedMLP` implementation
- Legacy educational implementations in `forgeNN.legacy`
- Complete activation function library
- Professional loss functions (cross-entropy, MSE)
- Vectorized optimizer with momentum support
- Comprehensive MNIST example achieving 93%+ accuracy
- Full documentation and performance guides
- PyPI-ready packaging configuration

### Features
- **Vectorized Operations**: NumPy-powered batch processing (100x+ speedup)
- **Dynamic Computation Graphs**: Automatic differentiation with gradient tracking
- **Complete Neural Networks**: From simple neurons to complex architectures
- **Production Loss Functions**: Cross-entropy, MSE with numerical stability
- **Educational Components**: Legacy implementations for learning purposes
- **High Performance**: 38,000+ samples/sec training speed

### Performance
- MNIST classification: 93%+ accuracy in under 2 seconds
- Training speed: 38,000+ samples per second
- Memory efficient vectorized operations
- Optimized backward pass implementations

### Documentation
- Comprehensive README with examples
- Performance optimization guide
- Activation function reference guide
- Complete installation instructions
- Full API documentation in docstrings

### Examples
- Complete MNIST classification demo
- Performance benchmarking examples
- Educational automatic differentiation examples
- Production-ready training loops

## [0.1.0] - 2025-09-01

### Added
- Initial development version
- Basic automatic differentiation engine
- Simple neural network implementations
- Educational examples and tutorials
