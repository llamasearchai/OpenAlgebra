cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(OpenAlgebra VERSION 1.0.0 LANGUAGES CXX)

# OpenAlgebra: High-Performance Sparse Linear Algebra Library
set(PROJECT_DESCRIPTION "Advanced sparse linear algebra library in Rust and C++")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(OPENALGEBRA_BUILD_TESTS "Build comprehensive test suite" OFF)
option(OPENALGEBRA_BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(OPENALGEBRA_BUILD_EXAMPLES "Build examples" OFF)
option(OPENALGEBRA_BUILD_PYTHON "Build Python bindings" ON)
option(OPENALGEBRA_ENABLE_CUDA "Enable CUDA acceleration" ON)
option(OPENALGEBRA_ENABLE_MPI "Enable MPI for distributed computing" ON)
option(OPENALGEBRA_ENABLE_OPENMP "Enable OpenMP parallelization" ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimization options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -DOPENALGEBRA_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /fp:fast")
endif()

# Find required packages
find_package(Threads REQUIRED)

# OpenMP for parallel processing
if(OPENALGEBRA_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DOPENALGEBRA_OPENMP_ENABLED)
        message(STATUS "OpenMP enabled for parallel processing")
    endif()
endif()

# MPI for distributed computing
if(OPENALGEBRA_ENABLE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        include_directories(${MPI_CXX_INCLUDE_DIRS})
        add_definitions(-DOPENALGEBRA_MPI_ENABLED)
        message(STATUS "MPI enabled for distributed computing")
    endif()
endif()

# CUDA for GPU acceleration
if(OPENALGEBRA_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
        add_definitions(-DOPENALGEBRA_CUDA_ENABLED)
        message(STATUS "CUDA enabled for GPU acceleration")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Core sparse linear algebra sources
set(OPENALGEBRA_CORE_SOURCES
    src/core/algebra/matrix/sparse_matrix.cpp
    src/core/algebra/matrix/matrix_operations.cpp
    src/core/algebra/tensor/sparse_tensor_coo.cpp
    src/core/algebra/tensor/tensor_operations.cpp
    src/core/algebra/solvers/iterative/cg_solver.cpp
    src/core/algebra/solvers/iterative/gmres_solver.cpp
    src/core/algebra/solvers/iterative/bicgstab_solver.cpp
    src/core/algebra/solvers/direct/multifrontal.cpp
    src/core/algebra/solvers/direct/supernodal.cpp
    src/core/algebra/solvers/preconditioners/amg_preconditioner.cpp
    src/core/algebra/solvers/preconditioners/ilu_preconditioner.cpp
    src/core/autograd/variable.cpp
    src/core/autograd/sparse_operations.cpp
    src/core/optimizers/sparse_sgd.cpp
    src/core/optimizers/sparse_adam.cpp
    src/core/utils/timer.cpp
    src/core/utils/logger.cpp
    src/core/utils/error_handling.cpp
)

# CUDA sources for GPU acceleration
if(OPENALGEBRA_ENABLE_CUDA)
    set(OPENALGEBRA_CUDA_SOURCES
        src/core/hardware/cuda/cuda_kernels.cu
        src/core/hardware/cuda/memory_manager.cu
        src/core/hardware/cuda/sparse_tensor_cuda.cu
    )
endif()

# MPI sources for distributed computing
if(OPENALGEBRA_ENABLE_MPI)
    set(OPENALGEBRA_MPI_SOURCES
        src/core/mpi/distributed_matrix.cpp
        src/core/mpi/distributed_tensor.cpp
        src/core/mpi/mpi_solver.cpp
    )
endif()

# Create main library
add_library(openalgebra 
    ${OPENALGEBRA_CORE_SOURCES}
    ${OPENALGEBRA_CUDA_SOURCES}
    ${OPENALGEBRA_MPI_SOURCES}
)

# Set library properties
set_target_properties(openalgebra PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Create interface library
add_library(openalgebra_interface INTERFACE)
target_include_directories(openalgebra_interface INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(openalgebra PUBLIC Threads::Threads)

if(OPENALGEBRA_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(openalgebra PUBLIC OpenMP::OpenMP_CXX)
endif()

if(OPENALGEBRA_ENABLE_MPI AND MPI_CXX_FOUND)
    target_link_libraries(openalgebra PUBLIC ${MPI_CXX_LIBRARIES})
endif()

if(OPENALGEBRA_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(openalgebra PUBLIC 
        CUDA::cusparse 
        CUDA::cublas 
        CUDA::cusolver
        CUDA::cudart
        CUDA::curand
        CUDA::cufft
    )
    
    # CUDA-specific properties
    set_property(TARGET openalgebra PROPERTY CUDA_RUNTIME_LIBRARY Static)
    set_property(TARGET openalgebra PROPERTY CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Install targets
install(TARGETS openalgebra openalgebra_interface
    EXPORT OpenAlgebraTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY src/
    DESTINATION include
    FILES_MATCHING 
    PATTERN "*.hpp" 
    PATTERN "*.cuh"
    PATTERN "*.h"
)

# Export targets
install(EXPORT OpenAlgebraTargets
    FILE OpenAlgebraTargets.cmake
    NAMESPACE OpenAlgebra::
    DESTINATION lib/cmake/OpenAlgebra
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    OpenAlgebraConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenAlgebraConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OpenAlgebraConfig.cmake
    INSTALL_DESTINATION lib/cmake/OpenAlgebra
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OpenAlgebraConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OpenAlgebraConfigVersion.cmake
    DESTINATION lib/cmake/OpenAlgebra
)

# Tests
if(OPENALGEBRA_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    set(TEST_SOURCES
        tests/unit/test_sparse_matrix.cpp
        tests/unit/test_tensors.cpp
        tests/unit/test_autograd.cpp
        tests/unit/test_solvers.cpp
        tests/integration/test_pipeline.cpp
    )
    
    if(OPENALGEBRA_ENABLE_CUDA)
        list(APPEND TEST_SOURCES 
            tests/unit/test_cuda_kernels.cpp
            tests/integration/test_gpu_acceleration.cpp
        )
    endif()
    
    if(OPENALGEBRA_ENABLE_MPI)
        list(APPEND TEST_SOURCES 
            tests/unit/test_mpi_operations.cpp
            tests/integration/test_distributed_computing.cpp
        )
    endif()
    
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        
        target_link_libraries(${test_name} 
            openalgebra 
            GTest::GTest 
            GTest::Main
        )
        
        # Add test-specific definitions
        target_compile_definitions(${test_name} PRIVATE 
            OPENALGEBRA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        )
        
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
    
endif()

# Benchmarks
if(OPENALGEBRA_BUILD_BENCHMARKS)
    add_executable(performance_suite 
        benchmarks/performance_suite.cpp
        benchmarks/accuracy_benchmarks.cpp
        benchmarks/datasets.cpp
    )
    target_link_libraries(performance_suite openalgebra)
    
    if(OPENALGEBRA_ENABLE_CUDA)
        add_executable(cuda_benchmarks benchmarks/cuda_benchmarks.cu)
        target_link_libraries(cuda_benchmarks openalgebra)
        set_property(TARGET cuda_benchmarks PROPERTY CUDA_RUNTIME_LIBRARY Static)
    endif()
    
    if(OPENALGEBRA_ENABLE_MPI)
        add_executable(distributed_scaling benchmarks/distributed_scaling.cpp)
        target_link_libraries(distributed_scaling openalgebra)
    endif()
endif()

# Examples
if(OPENALGEBRA_BUILD_EXAMPLES)
    add_executable(sparse_matrix_example examples/sparse_matrix_example.cpp)
    target_link_libraries(sparse_matrix_example openalgebra)
    
    add_executable(solver_example examples/solver_example.cpp)
    target_link_libraries(solver_example openalgebra)
    
    if(OPENALGEBRA_ENABLE_CUDA)
        add_executable(cuda_example examples/cuda_example.cu)
        target_link_libraries(cuda_example openalgebra)
        set_property(TARGET cuda_example PROPERTY CUDA_RUNTIME_LIBRARY Static)
    endif()
    
    if(OPENALGEBRA_ENABLE_MPI)
        add_executable(mpi_example examples/mpi_example.cpp)
        target_link_libraries(mpi_example openalgebra)
    endif()
endif()

# Python bindings
if(OPENALGEBRA_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
    
    pybind11_add_module(openalgebra_python 
        src/python/openalgebra_bindings.cpp
        src/python/sparse_matrix_bindings.cpp
        src/python/tensor_bindings.cpp
        src/python/solver_bindings.cpp
    )
    
    target_link_libraries(openalgebra_python PRIVATE openalgebra)
    
    set_target_properties(openalgebra_python PROPERTIES
        OUTPUT_NAME openalgebra
        SUFFIX ${PYTHON_MODULE_EXTENSION}
    )
    
    # Python utilities
    install(FILES
        src/python/__init__.py
        src/python/sparse_matrix.py
        src/python/autograd.py
        DESTINATION ${Python_SITEARCH}/openalgebra
    )
endif()

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "OpenAlgebra")
    set(DOXYGEN_PROJECT_BRIEF "High-Performance Sparse Linear Algebra Library")
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    set(DOXYGEN_INPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set(DOXYGEN_EXCLUDE_PATTERNS "*/internal/*")
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_CLASS_DIAGRAMS YES)
    set(DOXYGEN_HAVE_DOT YES)
    
    doxygen_add_docs(docs ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "OpenAlgebra")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance Sparse Linear Algebra Library")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "OpenAlgebra Team")
set(CPACK_PACKAGE_CONTACT "contact@openalgebra.org")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Platform-specific packaging
if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop;TGZ")
else()
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
    
    # Debian package configuration
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "OpenAlgebra Team")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libgomp1")
    
    # RPM package configuration
    set(CPACK_RPM_PACKAGE_GROUP "Applications/Science")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc, libstdc++, libgomp")
endif()

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "OpenAlgebra Configuration Summary:")
message(STATUS "=================================")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Core Features:")
message(STATUS "  OpenMP support: ${OPENALGEBRA_ENABLE_OPENMP}")
message(STATUS "  MPI support: ${OPENALGEBRA_ENABLE_MPI}")
message(STATUS "  CUDA support: ${OPENALGEBRA_ENABLE_CUDA}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Build tests: ${OPENALGEBRA_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${OPENALGEBRA_BUILD_BENCHMARKS}")
message(STATUS "  Build examples: ${OPENALGEBRA_BUILD_EXAMPLES}")
message(STATUS "  Build Python bindings: ${OPENALGEBRA_BUILD_PYTHON}")
message(STATUS "")
message(STATUS "Target Applications:")
message(STATUS "  - Sparse matrix computations")
message(STATUS "  - Iterative linear solvers")
message(STATUS "  - Scientific computing")
message(STATUS "  - Machine learning frameworks")
message(STATUS "  - High-performance computing")
message(STATUS "=================================")
message(STATUS "") 