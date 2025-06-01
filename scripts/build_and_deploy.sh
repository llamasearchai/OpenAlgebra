#!/bin/bash

# OpenAlgebra Medical AI Build and Deployment Script
# Production-ready build system for medical AI applications

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly BUILD_DIR="${PROJECT_ROOT}/build"
readonly INSTALL_DIR="${PROJECT_ROOT}/install"
readonly LOG_DIR="${PROJECT_ROOT}/logs"

# Build configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
NUM_CORES="${NUM_CORES:-$(nproc)}"
ENABLE_CUDA="${ENABLE_CUDA:-ON}"
ENABLE_MPI="${ENABLE_MPI:-ON}"
ENABLE_TESTS="${ENABLE_TESTS:-ON}"
ENABLE_MEDICAL_IO="${ENABLE_MEDICAL_IO:-ON}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Deployment configuration
DEPLOY_ENV="${DEPLOY_ENV:-development}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/llamasearchai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENABLE_GPU_DEPLOY="${ENABLE_GPU_DEPLOY:-true}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "${LOG_DIR}/build.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "${LOG_DIR}/build.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "${LOG_DIR}/build.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG_DIR}/build.log"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Build failed with exit code $exit_code"
        log_info "Check logs at ${LOG_DIR}/build.log for details"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Helper functions
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed"
        return 1
    fi
}

check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Essential build tools
    check_command cmake
    check_command ninja
    check_command gcc
    check_command g++
    check_command python3
    check_command pip3
    
    # Optional but recommended
    if ! check_command docker; then
        log_warning "Docker not found - container deployment will not be available"
    fi
    
    if [[ "$ENABLE_CUDA" == "ON" ]]; then
        if ! check_command nvcc; then
            log_warning "CUDA compiler not found - GPU acceleration will be disabled"
            ENABLE_CUDA="OFF"
        fi
    fi
    
    if [[ "$ENABLE_MPI" == "ON" ]]; then
        if ! check_command mpicc; then
            log_warning "MPI compiler not found - distributed computing will be disabled"
            ENABLE_MPI="OFF"
        fi
    fi
    
    log_success "Dependency check completed"
}

setup_directories() {
    log_info "Setting up build directories..."
    
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${PROJECT_ROOT}/data/dicom"
    mkdir -p "${PROJECT_ROOT}/data/nifti"
    mkdir -p "${PROJECT_ROOT}/models"
    mkdir -p "${PROJECT_ROOT}/results"
    
    log_success "Directories created"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
        python3 -m venv "${PROJECT_ROOT}/venv"
    fi
    
    # Activate virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Upgrade pip and install dependencies
    pip install --upgrade pip setuptools wheel
    pip install -r "${PROJECT_ROOT}/requirements.txt"
    
    # Install development dependencies if available
    if [[ -f "${PROJECT_ROOT}/requirements-dev.txt" ]]; then
        pip install -r "${PROJECT_ROOT}/requirements-dev.txt"
    fi
    
    log_success "Python dependencies installed"
}

configure_cmake() {
    log_info "Configuring CMake build system..."
    
    cd "${BUILD_DIR}"
    
    # CMake configuration options
    local cmake_args=(
        -G Ninja
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DOPENALGEBRA_BUILD_TESTS="${ENABLE_TESTS}"
        -DOPENALGEBRA_BUILD_BENCHMARKS=ON
        -DOPENALGEBRA_BUILD_EXAMPLES=ON
        -DOPENALGEBRA_BUILD_PYTHON=ON
        -DOPENALGEBRA_ENABLE_CUDA="${ENABLE_CUDA}"
        -DOPENALGEBRA_ENABLE_MPI="${ENABLE_MPI}"
        -DOPENALGEBRA_ENABLE_OPENMP=ON
        -DOPENALGEBRA_ENABLE_MEDICAL_IO="${ENABLE_MEDICAL_IO}"
        -DOPENALGEBRA_ENABLE_CLINICAL_VALIDATION=ON
    )
    
    # Add CUDA-specific options
    if [[ "$ENABLE_CUDA" == "ON" ]]; then
        cmake_args+=(
            -DCUDA_ARCHITECTURES="70;75;80;86"
            -DCMAKE_CUDA_COMPILER_ID=NVIDIA
        )
    fi
    
    # Add MPI-specific options
    if [[ "$ENABLE_MPI" == "ON" ]]; then
        cmake_args+=(
            -DMPI_CXX_COMPILER=mpicxx
            -DMPI_C_COMPILER=mpicc
        )
    fi
    
    # Configure with CMake
    cmake "${cmake_args[@]}" "${PROJECT_ROOT}"
    
    log_success "CMake configuration completed"
}

build_project() {
    log_info "Building OpenAlgebra Medical AI..."
    
    cd "${BUILD_DIR}"
    
    # Build with ninja
    ninja -j "${NUM_CORES}" 2>&1 | tee "${LOG_DIR}/build_output.log"
    
    log_success "Build completed successfully"
}

run_tests() {
    if [[ "$ENABLE_TESTS" == "ON" ]]; then
        log_info "Running test suite..."
        
        cd "${BUILD_DIR}"
        
        # Run C++ tests
        ctest --output-on-failure --parallel "${NUM_CORES}" 2>&1 | tee "${LOG_DIR}/test_output.log"
        
        # Run Python tests if available
        if [[ -d "${PROJECT_ROOT}/tests" ]]; then
            source "${PROJECT_ROOT}/venv/bin/activate"
            python -m pytest "${PROJECT_ROOT}/tests" -v --tb=short 2>&1 | tee "${LOG_DIR}/pytest_output.log"
        fi
        
        log_success "All tests passed"
    else
        log_info "Tests disabled - skipping test execution"
    fi
}

install_project() {
    log_info "Installing OpenAlgebra Medical AI..."
    
    cd "${BUILD_DIR}"
    ninja install
    
    # Install Python bindings
    if [[ -f "${BUILD_DIR}/python/setup.py" ]]; then
        source "${PROJECT_ROOT}/venv/bin/activate"
        cd "${BUILD_DIR}/python"
        pip install -e .
    fi
    
    log_success "Installation completed"
}

build_docker_images() {
    if command -v docker &> /dev/null; then
        log_info "Building Docker images..."
        
        cd "${PROJECT_ROOT}"
        
        # Build main production image
        docker build \
            --target production \
            --tag "${DOCKER_REGISTRY}/openalgebra-medical:${IMAGE_TAG}" \
            --tag "${DOCKER_REGISTRY}/openalgebra-medical:latest" \
            .
        
        # Build GPU-enabled image if CUDA is available
        if [[ "$ENABLE_GPU_DEPLOY" == "true" && "$ENABLE_CUDA" == "ON" ]]; then
            docker build \
                --target production \
                --tag "${DOCKER_REGISTRY}/openalgebra-medical:${IMAGE_TAG}-gpu" \
                --tag "${DOCKER_REGISTRY}/openalgebra-medical:latest-gpu" \
                --build-arg ENABLE_GPU=true \
                .
        fi
        
        log_success "Docker images built successfully"
    else
        log_warning "Docker not available - skipping image build"
    fi
}

deploy_development() {
    log_info "Deploying development environment..."
    
    cd "${PROJECT_ROOT}"
    
    # Start development services
    docker-compose -f docker-compose.yml up -d postgres redis
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Run database migrations if available
    if [[ -f "${PROJECT_ROOT}/migrations/init.sql" ]]; then
        docker-compose exec postgres psql -U openalgebra -d medical_ai -f /docker-entrypoint-initdb.d/init.sql
    fi
    
    log_success "Development environment deployed"
}

deploy_production() {
    log_info "Deploying production environment..."
    
    cd "${PROJECT_ROOT}"
    
    # Deploy full production stack
    docker-compose -f docker-compose.yml up -d
    
    # Wait for all services to be healthy
    log_info "Waiting for services to be healthy..."
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose ps | grep -q "unhealthy\|restarting"; then
            log_info "Services still starting... (attempt $((attempt + 1))/$max_attempts)"
            sleep 10
            ((attempt++))
        else
            break
        fi
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Some services failed to start properly"
        docker-compose ps
        return 1
    fi
    
    log_success "Production environment deployed successfully"
}

run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    cd "${BUILD_DIR}"
    
    # Run C++ benchmarks
    if [[ -f "${BUILD_DIR}/benchmarks/medical_performance_suite" ]]; then
        ./benchmarks/medical_performance_suite 2>&1 | tee "${LOG_DIR}/benchmark_output.log"
    fi
    
    # Run Python benchmarks
    if [[ -f "${PROJECT_ROOT}/benchmarks/python_benchmarks.py" ]]; then
        source "${PROJECT_ROOT}/venv/bin/activate"
        python "${PROJECT_ROOT}/benchmarks/python_benchmarks.py" 2>&1 | tee "${LOG_DIR}/python_benchmark_output.log"
    fi
    
    log_success "Benchmarks completed"
}

generate_documentation() {
    log_info "Generating documentation..."
    
    # Generate C++ documentation with Doxygen
    if command -v doxygen &> /dev/null && [[ -f "${PROJECT_ROOT}/Doxyfile" ]]; then
        cd "${PROJECT_ROOT}"
        doxygen Doxyfile
    fi
    
    # Generate Python documentation with Sphinx
    if [[ -d "${PROJECT_ROOT}/docs" ]]; then
        source "${PROJECT_ROOT}/venv/bin/activate"
        cd "${PROJECT_ROOT}/docs"
        make html
    fi
    
    log_success "Documentation generated"
}

package_release() {
    log_info "Packaging release artifacts..."
    
    local release_dir="${PROJECT_ROOT}/release"
    mkdir -p "${release_dir}"
    
    # Package binaries
    cd "${INSTALL_DIR}"
    tar -czf "${release_dir}/openalgebra-medical-${IMAGE_TAG}-$(uname -m).tar.gz" .
    
    # Package Python wheels
    if [[ -d "${BUILD_DIR}/python/dist" ]]; then
        cp "${BUILD_DIR}/python/dist"/*.whl "${release_dir}/"
    fi
    
    # Package Docker images
    if command -v docker &> /dev/null; then
        docker save "${DOCKER_REGISTRY}/openalgebra-medical:${IMAGE_TAG}" | gzip > "${release_dir}/openalgebra-medical-${IMAGE_TAG}.tar.gz"
    fi
    
    # Generate checksums
    cd "${release_dir}"
    sha256sum * > checksums.sha256
    
    log_success "Release artifacts packaged in ${release_dir}"
}

validate_deployment() {
    log_info "Validating deployment..."
    
    # Health check API endpoints
    local api_url="http://localhost:8000"
    local max_attempts=10
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f "${api_url}/health" &> /dev/null; then
            log_success "API health check passed"
            break
        else
            log_info "API not ready yet... (attempt $((attempt + 1))/$max_attempts)"
            sleep 5
            ((attempt++))
        fi
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "API health check failed"
        return 1
    fi
    
    # Test medical AI endpoints
    if curl -f "${api_url}/medical/health" &> /dev/null; then
        log_success "Medical AI endpoints accessible"
    else
        log_warning "Medical AI endpoints not accessible"
    fi
    
    # Validate database connectivity
    if docker-compose exec postgres pg_isready -U openalgebra -d medical_ai &> /dev/null; then
        log_success "Database connectivity validated"
    else
        log_error "Database connectivity failed"
        return 1
    fi
    
    log_success "Deployment validation completed"
}

print_usage() {
    cat << EOF
OpenAlgebra Medical AI Build and Deployment Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    build       Build the project (default)
    test        Run tests only
    install     Install the project
    docker      Build Docker images
    deploy      Deploy environment
    benchmark   Run performance benchmarks
    docs        Generate documentation
    package     Package release artifacts
    validate    Validate deployment
    clean       Clean build artifacts
    all         Run complete build, test, and deploy pipeline

Options:
    -h, --help              Show this help message
    -t, --build-type TYPE   Build type: Debug, Release, RelWithDebInfo (default: Release)
    -j, --jobs NUM          Number of parallel jobs (default: number of CPU cores)
    --cuda                  Enable CUDA support (default: ON)
    --no-cuda               Disable CUDA support
    --mpi                   Enable MPI support (default: ON)
    --no-mpi                Disable MPI support
    --no-tests              Disable tests
    --env ENV               Deployment environment: development, production (default: development)
    --tag TAG               Docker image tag (default: latest)
    --registry URL          Docker registry URL
    --verbose               Enable verbose output

Environment Variables:
    BUILD_TYPE              Build type (Debug, Release, RelWithDebInfo)
    NUM_CORES               Number of parallel build jobs
    ENABLE_CUDA             Enable CUDA support (ON/OFF)
    ENABLE_MPI              Enable MPI support (ON/OFF)
    ENABLE_TESTS            Enable tests (ON/OFF)
    DEPLOY_ENV              Deployment environment
    DOCKER_REGISTRY         Docker registry URL
    IMAGE_TAG               Docker image tag

Examples:
    $0                      # Build with default settings
    $0 --build-type Debug   # Debug build
    $0 deploy --env production  # Deploy production environment
    $0 all --cuda --tag v1.0   # Complete pipeline with CUDA and custom tag
    $0 clean                # Clean build artifacts

EOF
}

# Parse command line arguments
COMMAND="build"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -t|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_CORES="$2"
            shift 2
            ;;
        --cuda)
            ENABLE_CUDA="ON"
            shift
            ;;
        --no-cuda)
            ENABLE_CUDA="OFF"
            shift
            ;;
        --mpi)
            ENABLE_MPI="ON"
            shift
            ;;
        --no-mpi)
            ENABLE_MPI="OFF"
            shift
            ;;
        --no-tests)
            ENABLE_TESTS="OFF"
            shift
            ;;
        --env)
            DEPLOY_ENV="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        build|test|install|docker|deploy|benchmark|docs|package|validate|clean|all)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Enable verbose output if requested
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Main execution
main() {
    log_info "Starting OpenAlgebra Medical AI build and deployment"
    log_info "Command: $COMMAND"
    log_info "Build type: $BUILD_TYPE"
    log_info "CUDA enabled: $ENABLE_CUDA"
    log_info "MPI enabled: $ENABLE_MPI"
    log_info "Tests enabled: $ENABLE_TESTS"
    log_info "Environment: $DEPLOY_ENV"
    
    setup_directories
    
    case "$COMMAND" in
        build)
            check_dependencies
            install_python_dependencies
            configure_cmake
            build_project
            ;;
        test)
            run_tests
            ;;
        install)
            check_dependencies
            install_python_dependencies
            configure_cmake
            build_project
            run_tests
            install_project
            ;;
        docker)
            build_docker_images
            ;;
        deploy)
            if [[ "$DEPLOY_ENV" == "production" ]]; then
                deploy_production
            else
                deploy_development
            fi
            validate_deployment
            ;;
        benchmark)
            run_benchmarks
            ;;
        docs)
            generate_documentation
            ;;
        package)
            package_release
            ;;
        validate)
            validate_deployment
            ;;
        clean)
            log_info "Cleaning build artifacts..."
            rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
            docker system prune -f
            log_success "Clean completed"
            ;;
        all)
            check_dependencies
            install_python_dependencies
            configure_cmake
            build_project
            run_tests
            install_project
            build_docker_images
            if [[ "$DEPLOY_ENV" == "production" ]]; then
                deploy_production
            else
                deploy_development
            fi
            validate_deployment
            run_benchmarks
            generate_documentation
            package_release
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            print_usage
            exit 1
            ;;
    esac
    
    log_success "OpenAlgebra Medical AI $COMMAND completed successfully!"
    log_info "Logs available at: ${LOG_DIR}"
}

# Run main function
main "$@" 