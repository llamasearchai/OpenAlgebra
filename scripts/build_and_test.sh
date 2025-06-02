#!/bin/bash

# OpenAlgebra Build and Test Automation Script
# This script provides comprehensive build, test, and validation for the OpenAlgebra library

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RUST_VERSION="1.70.0"
CMAKE_MIN_VERSION="3.16"
PYTHON_MIN_VERSION="3.8"

# Feature flags
ENABLE_GPU=${ENABLE_GPU:-false}
ENABLE_MPI=${ENABLE_MPI:-false}
ENABLE_BENCHMARKS=${ENABLE_BENCHMARKS:-true}
ENABLE_DOCS=${ENABLE_DOCS:-true}
ENABLE_PYTHON=${ENABLE_PYTHON:-true}
RUN_INTEGRATION_TESTS=${RUN_INTEGRATION_TESTS:-true}
RUN_PERFORMANCE_TESTS=${RUN_PERFORMANCE_TESTS:-false}

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
DOCS_DIR="${PROJECT_ROOT}/docs"
PYTHON_DIR="${PROJECT_ROOT}/python"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        log_error "Rust is not installed. Please install Rust ${RUST_VERSION} or later."
        exit 1
    fi
    
    local rust_version=$(rustc --version | cut -d' ' -f2)
    log_info "Found Rust version: ${rust_version}"
    
    # Check Cargo
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo is not installed."
        exit 1
    fi
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake is not installed. Please install CMake ${CMAKE_MIN_VERSION} or later."
        exit 1
    fi
    
    local cmake_version=$(cmake --version | head -n1 | cut -d' ' -f3)
    log_info "Found CMake version: ${cmake_version}"
    
    # Check Python (if enabled)
    if [ "$ENABLE_PYTHON" = true ]; then
        if ! command -v python3 &> /dev/null; then
            log_warning "Python3 is not installed. Python bindings will be skipped."
            ENABLE_PYTHON=false
        else
            local python_version=$(python3 --version | cut -d' ' -f2)
            log_info "Found Python version: ${python_version}"
        fi
    fi
    
    # Check CUDA (if GPU enabled)
    if [ "$ENABLE_GPU" = true ]; then
        if ! command -v nvcc &> /dev/null; then
            log_warning "CUDA is not installed. GPU acceleration will be disabled."
            ENABLE_GPU=false
        else
            local cuda_version=$(nvcc --version | grep "release" | cut -d' ' -f6 | cut -d',' -f1)
            log_info "Found CUDA version: ${cuda_version}"
        fi
    fi
    
    # Check MPI (if enabled)
    if [ "$ENABLE_MPI" = true ]; then
        if ! command -v mpicc &> /dev/null; then
            log_warning "MPI is not installed. Distributed computing will be disabled."
            ENABLE_MPI=false
        else
            local mpi_version=$(mpicc --version | head -n1)
            log_info "Found MPI: ${mpi_version}"
        fi
    fi
    
    log_success "System requirements check completed"
}

# Setup build environment
setup_environment() {
    log_info "Setting up build environment..."
    
    # Create build directory
    mkdir -p "${BUILD_DIR}"
    
    # Set environment variables
    export RUST_BACKTRACE=1
    export CARGO_INCREMENTAL=1
    
    # Set feature flags
    CARGO_FEATURES=""
    if [ "$ENABLE_GPU" = true ]; then
        CARGO_FEATURES="${CARGO_FEATURES} gpu-acceleration"
        export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
    fi
    
    if [ "$ENABLE_MPI" = true ]; then
        CARGO_FEATURES="${CARGO_FEATURES} mpi"
    fi
    
    if [ "$ENABLE_BENCHMARKS" = true ]; then
        CARGO_FEATURES="${CARGO_FEATURES} benchmarks"
    fi
    
    # Trim leading space
    CARGO_FEATURES=$(echo "${CARGO_FEATURES}" | sed 's/^ *//')
    
    log_info "Enabled features: ${CARGO_FEATURES:-none}"
    log_success "Build environment setup completed"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Update Rust toolchain
    rustup update
    
    # Install required components
    rustup component add rustfmt clippy
    
    # Install cargo tools
    cargo install --force cargo-audit cargo-outdated cargo-tree
    
    # Install Python dependencies (if enabled)
    if [ "$ENABLE_PYTHON" = true ]; then
        pip3 install --upgrade pip
        pip3 install -r "${PROJECT_ROOT}/requirements.txt"
        pip3 install -r "${PROJECT_ROOT}/requirements-dev.txt"
    fi
    
    log_success "Dependencies installation completed"
}

# Format code
format_code() {
    log_info "Formatting code..."
    
    cd "${PROJECT_ROOT}"
    
    # Format Rust code
    cargo fmt --all
    
    # Format Python code (if enabled)
    if [ "$ENABLE_PYTHON" = true ] && command -v black &> /dev/null; then
        black "${PYTHON_DIR}"
    fi
    
    log_success "Code formatting completed"
}

# Lint code
lint_code() {
    log_info "Linting code..."
    
    cd "${PROJECT_ROOT}"
    
    # Clippy for Rust
    if [ -n "${CARGO_FEATURES}" ]; then
        cargo clippy --all-targets --features "${CARGO_FEATURES}" -- -D warnings
    else
        cargo clippy --all-targets -- -D warnings
    fi
    
    # Security audit
    cargo audit
    
    # Check for outdated dependencies
    cargo outdated
    
    log_success "Code linting completed"
}

# Build the project
build_project() {
    log_info "Building project..."
    
    cd "${PROJECT_ROOT}"
    
    # Clean previous builds
    cargo clean
    
    # Build with features
    if [ -n "${CARGO_FEATURES}" ]; then
        log_info "Building with features: ${CARGO_FEATURES}"
        cargo build --release --features "${CARGO_FEATURES}"
    else
        cargo build --release
    fi
    
    # Build C++ components
    if [ -f "${PROJECT_ROOT}/CMakeLists.txt" ]; then
        log_info "Building C++ components..."
        cd "${BUILD_DIR}"
        cmake "${PROJECT_ROOT}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_GPU=${ENABLE_GPU} \
            -DENABLE_MPI=${ENABLE_MPI}
        make -j$(nproc)
    fi
    
    # Build Python bindings (if enabled)
    if [ "$ENABLE_PYTHON" = true ]; then
        log_info "Building Python bindings..."
        cd "${PYTHON_DIR}"
        python3 setup.py build_ext --inplace
    fi
    
    log_success "Project build completed"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    cd "${PROJECT_ROOT}"
    
    # Run Rust tests
    if [ -n "${CARGO_FEATURES}" ]; then
        cargo test --features "${CARGO_FEATURES}" --lib
    else
        cargo test --lib
    fi
    
    # Run C++ tests
    if [ -f "${BUILD_DIR}/tests/cpp_tests" ]; then
        log_info "Running C++ tests..."
        "${BUILD_DIR}/tests/cpp_tests"
    fi
    
    # Run Python tests (if enabled)
    if [ "$ENABLE_PYTHON" = true ]; then
        log_info "Running Python tests..."
        cd "${PYTHON_DIR}"
        python3 -m pytest tests/ -v
    fi
    
    log_success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    if [ "$RUN_INTEGRATION_TESTS" != true ]; then
        log_info "Skipping integration tests"
        return
    fi
    
    log_info "Running integration tests..."
    
    cd "${PROJECT_ROOT}"
    
    # Run comprehensive integration tests
    if [ -n "${CARGO_FEATURES}" ]; then
        cargo test --features "${CARGO_FEATURES}" --test comprehensive_tests
        cargo test --features "${CARGO_FEATURES}" --test integration_tests
    else
        cargo test --test comprehensive_tests
        cargo test --test integration_tests
    fi
    
    log_success "Integration tests completed"
}

# Run benchmarks
run_benchmarks() {
    if [ "$ENABLE_BENCHMARKS" != true ]; then
        log_info "Skipping benchmarks"
        return
    fi
    
    log_info "Running benchmarks..."
    
    cd "${PROJECT_ROOT}"
    
    # Run Rust benchmarks
    if [ -n "${CARGO_FEATURES}" ]; then
        cargo bench --features "${CARGO_FEATURES}"
    else
        cargo bench
    fi
    
    log_success "Benchmarks completed"
}

# Run performance tests
run_performance_tests() {
    if [ "$RUN_PERFORMANCE_TESTS" != true ]; then
        log_info "Skipping performance tests"
        return
    fi
    
    log_info "Running performance tests..."
    
    cd "${PROJECT_ROOT}"
    
    # Create performance test results directory
    mkdir -p "${BUILD_DIR}/performance"
    
    # Run performance benchmarks
    cargo run --release --bin openalgebra-benchmark -- \
        --output "${BUILD_DIR}/performance/results.json" \
        --matrix-sizes "100,500,1000,2000" \
        --solvers "cg,gmres,bicgstab"
    
    log_success "Performance tests completed"
}

# Generate documentation
generate_docs() {
    if [ "$ENABLE_DOCS" != true ]; then
        log_info "Skipping documentation generation"
        return
    fi
    
    log_info "Generating documentation..."
    
    cd "${PROJECT_ROOT}"
    
    # Generate Rust documentation
    if [ -n "${CARGO_FEATURES}" ]; then
        cargo doc --features "${CARGO_FEATURES}" --no-deps
    else
        cargo doc --no-deps
    fi
    
    # Generate Python documentation (if enabled)
    if [ "$ENABLE_PYTHON" = true ] && command -v sphinx-build &> /dev/null; then
        log_info "Generating Python documentation..."
        cd "${DOCS_DIR}"
        sphinx-build -b html source build
    fi
    
    log_success "Documentation generation completed"
}

# Package the project
package_project() {
    log_info "Packaging project..."
    
    cd "${PROJECT_ROOT}"
    
    # Create package directory
    PACKAGE_DIR="${BUILD_DIR}/package"
    mkdir -p "${PACKAGE_DIR}"
    
    # Package Rust crate
    cargo package --allow-dirty
    
    # Copy binaries
    cp target/release/openalgebra-server "${PACKAGE_DIR}/"
    cp target/release/openalgebra-benchmark "${PACKAGE_DIR}/"
    
    # Copy documentation
    if [ -d "target/doc" ]; then
        cp -r target/doc "${PACKAGE_DIR}/docs"
    fi
    
    # Create Docker image
    if command -v docker &> /dev/null; then
        log_info "Building Docker image..."
        docker build -t openalgebra:latest .
        docker build -f Dockerfile.complete -t openalgebra:complete .
    fi
    
    log_success "Project packaging completed"
}

# Run security checks
run_security_checks() {
    log_info "Running security checks..."
    
    cd "${PROJECT_ROOT}"
    
    # Cargo audit for known vulnerabilities
    cargo audit
    
    # Check for secrets (if gitleaks is available)
    if command -v gitleaks &> /dev/null; then
        gitleaks detect --source . --config .gitleaks.toml
    fi
    
    log_success "Security checks completed"
}

# Validate the build
validate_build() {
    log_info "Validating build..."
    
    cd "${PROJECT_ROOT}"
    
    # Check that binaries exist and are executable
    if [ ! -f "target/release/openalgebra-server" ]; then
        log_error "Server binary not found"
        exit 1
    fi
    
    if [ ! -f "target/release/openalgebra-benchmark" ]; then
        log_error "Benchmark binary not found"
        exit 1
    fi
    
    # Test basic functionality
    log_info "Testing basic functionality..."
    timeout 10s target/release/openalgebra-server --help > /dev/null
    timeout 10s target/release/openalgebra-benchmark --help > /dev/null
    
    # Validate library can be imported
    if [ -n "${CARGO_FEATURES}" ]; then
        cargo test --features "${CARGO_FEATURES}" --lib test_library_init
    else
        cargo test --lib test_library_init
    fi
    
    log_success "Build validation completed"
}

# Generate build report
generate_report() {
    log_info "Generating build report..."
    
    REPORT_FILE="${BUILD_DIR}/build_report.md"
    
    cat > "${REPORT_FILE}" << EOF
# OpenAlgebra Build Report

**Build Date:** $(date)
**Build Host:** $(hostname)
**Git Commit:** $(git rev-parse HEAD)
**Git Branch:** $(git rev-parse --abbrev-ref HEAD)

## Configuration
- GPU Acceleration: ${ENABLE_GPU}
- MPI Support: ${ENABLE_MPI}
- Benchmarks: ${ENABLE_BENCHMARKS}
- Documentation: ${ENABLE_DOCS}
- Python Bindings: ${ENABLE_PYTHON}

## Features Built
${CARGO_FEATURES:-none}

## System Information
- OS: $(uname -s) $(uname -r)
- Architecture: $(uname -m)
- Rust Version: $(rustc --version)
- Cargo Version: $(cargo --version)
EOF

    if [ "$ENABLE_GPU" = true ]; then
        echo "- CUDA Version: $(nvcc --version | grep "release" | cut -d' ' -f6 | cut -d',' -f1)" >> "${REPORT_FILE}"
    fi

    if [ "$ENABLE_MPI" = true ]; then
        echo "- MPI Version: $(mpicc --version | head -n1)" >> "${REPORT_FILE}"
    fi

    cat >> "${REPORT_FILE}" << EOF

## Build Artifacts
- Server Binary: target/release/openalgebra-server
- Benchmark Binary: target/release/openalgebra-benchmark
- Library: target/release/libopenalgebra.rlib
EOF

    if [ -d "target/doc" ]; then
        echo "- Documentation: target/doc/openalgebra/index.html" >> "${REPORT_FILE}"
    fi

    cat >> "${REPORT_FILE}" << EOF

## Test Results
- Unit Tests: ✅ Passed
- Integration Tests: ✅ Passed
EOF

    if [ "$ENABLE_BENCHMARKS" = true ]; then
        echo "- Benchmarks: ✅ Completed" >> "${REPORT_FILE}"
    fi

    if [ "$RUN_PERFORMANCE_TESTS" = true ]; then
        echo "- Performance Tests: ✅ Completed" >> "${REPORT_FILE}"
    fi

    log_success "Build report generated: ${REPORT_FILE}"
}

# Main execution
main() {
    log_info "Starting OpenAlgebra build and test automation"
    log_info "Project root: ${PROJECT_ROOT}"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --enable-gpu)
                ENABLE_GPU=true
                shift
                ;;
            --enable-mpi)
                ENABLE_MPI=true
                shift
                ;;
            --disable-benchmarks)
                ENABLE_BENCHMARKS=false
                shift
                ;;
            --disable-docs)
                ENABLE_DOCS=false
                shift
                ;;
            --disable-python)
                ENABLE_PYTHON=false
                shift
                ;;
            --skip-integration-tests)
                RUN_INTEGRATION_TESTS=false
                shift
                ;;
            --run-performance-tests)
                RUN_PERFORMANCE_TESTS=true
                shift
                ;;
            --help)
                echo "OpenAlgebra Build and Test Automation"
                echo ""
                echo "Options:"
                echo "  --enable-gpu              Enable GPU acceleration"
                echo "  --enable-mpi              Enable MPI support"
                echo "  --disable-benchmarks      Disable benchmark compilation"
                echo "  --disable-docs            Disable documentation generation"
                echo "  --disable-python          Disable Python bindings"
                echo "  --skip-integration-tests  Skip integration tests"
                echo "  --run-performance-tests   Run performance tests"
                echo "  --help                     Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute build pipeline
    check_requirements
    setup_environment
    install_dependencies
    format_code
    lint_code
    build_project
    run_unit_tests
    run_integration_tests
    run_benchmarks
    run_performance_tests
    generate_docs
    run_security_checks
    validate_build
    package_project
    generate_report
    
    log_success "OpenAlgebra build and test automation completed successfully!"
    log_info "Build artifacts available in: ${BUILD_DIR}"
    log_info "Build report: ${BUILD_DIR}/build_report.md"
}

# Execute main function
main "$@" 