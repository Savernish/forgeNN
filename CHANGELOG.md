# Changelog

All notable changes to forgeNN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
