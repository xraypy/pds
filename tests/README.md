# Tests for PDS

This directory contains pytest-based tests for the PDS codebase and is structured as a Python package.

## Structure

- `__init__.py` - Package initialization, shared test utilities, and test runner
- `test_active_area.py` - Tests for the active area calculation module
- `test_background.py` - Tests for the background determination module
- `test_file_locker.py` - Tests for the cross-platform file locking module  
- `test_image_data.py` - Tests for the image data handling module
- `test_filtertools.py` - Tests for the HDF5 filtering and date/list operations module
- `test_lattice.py` - Tests for the crystallographic lattice calculations module
- `test_mathutil.py` - Tests for mathematical utility functions and numpy/scipy wrappers
- `test_polygon.py` - Tests for generalized polygon computations and geometric operations

## Running Tests

To run all tests using the PDS CLI:
```bash
python pds.py -t
```

To run all tests using the test package directly:
```bash
python -c "from tests import run_all_tests; run_all_tests()"
```

To run all tests using pytest directly:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_active_area.py
pytest tests/test_background.py
pytest tests/test_file_locker.py
pytest tests/test_image_data.py
pytest tests/test_filtertools.py
pytest tests/test_lattice.py
pytest tests/test_mathutil.py
pytest tests/test_polygon.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

## Test Categories

- **Compatibility Tests**: Ensure Python 2/3 compatibility across all modules
- **Functional Tests**: Test core functionality across all modules
- **Edge Case Tests**: Test boundary conditions and error handling
- **Mathematical Precision**: Validate numerical accuracy and algorithm behavior
- **Integration Tests**: Test module interactions
- **Concurrency Tests**: Test thread-safe operations (in `test_file_locker.py`)
- **Cross-platform Tests**: Test OS-specific functionality (file locking, path handling)
- **Performance Tests**: Basic performance benchmarks

## Special Test Features

### Active Area Calculations (`test_active_area.py`)
- **Surface Coordinate Transformation**: Lab frame to surface frame matrix calculations
- **Geometric Projections**: Beam, detector, and sample polygon projections onto surface
- **Area Calculations**: Round and polygon sample area computations
- **Surface Intercepts**: Vector projection calculations with boundary constraints
- **Python 2 Compatibility**: Validates updated Python 3 code against original behavior
- **Error Handling**: Invalid geometry specifications and edge cases
- **Visualization Testing**: Matplotlib integration for debugging plots
- **Type Safety**: Comprehensive type annotation validation

### Mathematical Utilities (`test_mathutil.py`)
- **Numpy Wrappers**: Tests for `ave()`, `std()` with deprecated `numpy.ave()` replacement
- **Trigonometric Functions**: Degree-based trig functions with precise validation
- **Vector Operations**: Cartesian magnitude and angle calculations
- **Optimization Wrapper**: scipy.optimize.leastsq wrapper with error handling
- **Random Distributions**: Comprehensive random number generation testing
- **Python 2 Compatibility**: Validates modernized code against original behavior

### Polynomial Background (`test_background.py`)
- **Kajfosz-Kwiatek Algorithm**: Comprehensive background determination testing
- **Array Operations**: Compression and expansion with various factors
- **Mathematical Precision**: Numerical accuracy validation
- **Edge Cases**: Boundary conditions and error handling

### HDF5 Compatibility (`test_image_data.py`)
- Tests for Python 2→3 migration string/bytes compatibility
- Unicode handling in PyTables group names
- Comprehensive data type conversion testing

### File Locking (`test_file_locker.py`)
- Concurrent access testing with thread synchronization
- Cross-platform username/hostname detection
- Context manager and manual lock/release patterns

### Polygon Computations (`test_polygon.py`)
- **Line Operations**: Line parameter calculation and intersection algorithms  
- **Point-in-Polygon**: Testing various polygon shapes and edge cases
- **Polygon Intersection**: Inner polygon calculation and overlap detection
- **Area Calculations**: Both analytical and numerical integration methods
- **Coordinate Transformations**: Rotation and scaling operations
- **Sorting Algorithms**: Point sorting by angular position
- **Python 2 Compatibility**: Exact preservation of original computational behavior
- **Plotting Integration**: Matplotlib integration for visualization testing
- **Edge Cases**: Degenerate polygons, very small/large coordinate ranges

### HDF5 Filtering (`test_filtertools.py`)
- Tests for Python 2→3 migration with original behavior preservation
- Date parsing and comparison functions (is_before, is_after)
- List operations (union, intersection) with None value handling
- HDF5 scan filtering with string/bytes compatibility
- Point-level and scan-level filtering validation
- Numeric type compatibility (numpy integers, floats)
- Original Python 2 eval() expression compatibility
- Timeout and error condition handling

### Crystallographic Lattice (`test_lattice.py`)
- **Unit Cell Parameters**: Lattice parameter initialization and updates
- **Metric Tensor Calculations**: Real and reciprocal space metric tensors
- **Volume Calculations**: Unit cell volume in real and reciprocal space
- **Vector Operations**: Dot products, magnitudes, and angles using metric tensors
- **Diffraction Calculations**: d-spacing and 2θ angle calculations
- **Coordinate Transformations**: Real ↔ reciprocal space transformations
- **Lattice Transforms**: Basis rotations, shifts, and coordinate system changes
- **Original Test Functions**: Exact reproduction of Python 2 test_lattice() and test_transform()
- **Crystallographic Systems**: Cubic, hexagonal, and rhombohedral lattice testing
- **Mathematical Precision**: High-precision validation of crystallographic calculations
- **Type Annotations**: Full Python 3.13 type hint compatibility
- **Edge Cases**: Zero parameters, extreme angles, and numerical stability
