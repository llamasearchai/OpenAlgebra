cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(OpenAlgebra VERSION 1.0.0 LANGUAGES CXX CUDA)

# OpenAlgebra: High-Performance Sparse Linear Algebra for Medical AI Model Development
set(PROJECT_DESCRIPTION "Advanced sparse linear algebra library optimized for medical AI applications")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(OPENALGEBRA_BUILD_TESTS "Build comprehensive test suite" ON)
option(OPENALGEBRA_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(OPENALGEBRA_BUILD_EXAMPLES "Build medical AI examples" ON)
option(OPENALGEBRA_BUILD_PYTHON "Build Python bindings with medical AI support" ON)
option(OPENALGEBRA_ENABLE_CUDA "Enable CUDA acceleration for medical imaging" ON)
option(OPENALGEBRA_ENABLE_MPI "Enable MPI for distributed medical computing" ON)
option(OPENALGEBRA_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(OPENALGEBRA_ENABLE_MEDICAL_IO "Enable medical imaging I/O (DICOM, NIfTI)" ON)
option(OPENALGEBRA_ENABLE_CLINICAL_VALIDATION "Enable clinical validation tools" ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options for medical AI optimization
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

# OpenMP for parallel medical image processing
if(OPENALGEBRA_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DOPENALGEBRA_OPENMP_ENABLED)
        message(STATUS "OpenMP enabled for parallel medical image processing")
    endif()
endif()

# MPI for distributed medical computing
if(OPENALGEBRA_ENABLE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        include_directories(${MPI_CXX_INCLUDE_DIRS})
        add_definitions(-DOPENALGEBRA_MPI_ENABLED)
        message(STATUS "MPI enabled for distributed medical computing")
    endif()
endif()

# CUDA for GPU-accelerated medical imaging
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
        message(STATUS "CUDA enabled for GPU-accelerated medical imaging")
    endif()
endif()

# Medical imaging I/O libraries
if(OPENALGEBRA_ENABLE_MEDICAL_IO)
    # DCMTK for DICOM support
    find_package(DCMTK QUIET)
    if(DCMTK_FOUND)
        add_definitions(-DOPENALGEBRA_DCMTK_ENABLED)
        message(STATUS "DCMTK found - DICOM support enabled")
    else()
        message(WARNING "DCMTK not found - DICOM support disabled")
    endif()
    
    # ITK for medical image processing
    find_package(ITK QUIET)
    if(ITK_FOUND)
        include(${ITK_USE_FILE})
        add_definitions(-DOPENALGEBRA_ITK_ENABLED)
        message(STATUS "ITK found - Advanced medical image processing enabled")
    else()
        message(WARNING "ITK not found - Advanced medical image processing disabled")
    endif()
    
    # VTK for medical visualization
    find_package(VTK QUIET)
    if(VTK_FOUND)
        add_definitions(-DOPENALGEBRA_VTK_ENABLED)
        message(STATUS "VTK found - Medical visualization enabled")
    else()
        message(WARNING "VTK not found - Medical visualization disabled")
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
    src/core/algebra/solvers/preconditioners/medical_preconditioners.cpp
    src/core/autograd/variable.cpp
    src/core/autograd/sparse_operations.cpp
    src/core/optimizers/sparse_sgd.cpp
    src/core/optimizers/sparse_adam.cpp
    src/core/optimizers/medical_optimizers.cpp
    src/core/utils/timer.cpp
    src/core/utils/logger.cpp
    src/core/utils/error_handling.cpp
    src/core/utils/medical_utils.cpp
)

# CUDA sources for medical imaging acceleration
if(OPENALGEBRA_ENABLE_CUDA)
    set(OPENALGEBRA_CUDA_SOURCES
        src/core/hardware/cuda/cuda_kernels.cu
        src/core/hardware/cuda/medical_cuda_ops.cu
        src/core/hardware/cuda/memory_manager.cu
        src/core/hardware/cuda/sparse_tensor_cuda.cu
    )
endif()

# MPI sources for distributed medical computing
if(OPENALGEBRA_ENABLE_MPI)
    set(OPENALGEBRA_MPI_SOURCES
        src/core/mpi/distributed_matrix.cpp
        src/core/mpi/distributed_tensor.cpp
        src/core/mpi/mpi_medical_solver.cpp
        src/core/mpi/federated_learning.cpp
    )
endif()

# Medical AI specific sources
set(OPENALGEBRA_MEDICAL_SOURCES
    src/medical_ai/data_io/dicom_processor.cpp
    src/medical_ai/data_io/nifti_processor.cpp
    src/medical_ai/data_io/medical_image_converter.cpp
    src/medical_ai/data_io/sparse_feature_extractor.cpp
    src/medical_ai/models/sparse_cnn.cpp
    src/medical_ai/models/medical_transformer.cpp
    src/medical_ai/models/graph_neural_networks.cpp
    src/medical_ai/models/federated_learning.cpp
    src/medical_ai/preprocessing/medical_normalization.cpp
    src/medical_ai/preprocessing/roi_extraction.cpp
    src/medical_ai/preprocessing/data_augmentation.cpp
    src/medical_ai/evaluation/medical_metrics.cpp
    src/medical_ai/evaluation/cross_validation.cpp
    src/medical_ai/evaluation/clinical_validation.cpp
)

# Create main library
add_library(openalgebra 
    ${OPENALGEBRA_CORE_SOURCES}
    ${OPENALGEBRA_CUDA_SOURCES}
    ${OPENALGEBRA_MPI_SOURCES}
    ${OPENALGEBRA_MEDICAL_SOURCES}
)

# Set library properties
set_target_properties(openalgebra PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Create interface for medical AI applications
add_library(openalgebra_medical INTERFACE)
target_include_directories(openalgebra_medical INTERFACE
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

# Medical imaging libraries
if(OPENALGEBRA_ENABLE_MEDICAL_IO)
    if(DCMTK_FOUND)
        target_link_libraries(openalgebra PUBLIC ${DCMTK_LIBRARIES})
        target_include_directories(openalgebra PUBLIC ${DCMTK_INCLUDE_DIRS})
    endif()
    
    if(ITK_FOUND)
        target_link_libraries(openalgebra PUBLIC ${ITK_LIBRARIES})
    endif()
    
    if(VTK_FOUND)
        target_link_libraries(openalgebra PUBLIC ${VTK_LIBRARIES})
    endif()
endif()

# Install targets
install(TARGETS openalgebra openalgebra_medical
    EXPORT OpenAlgebraTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers with medical AI structure
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

# Medical AI specific tests
if(OPENALGEBRA_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    set(TEST_SOURCES
        tests/unit/test_sparse_matrix.cpp
        tests/unit/test_medical_tensors.cpp
        tests/unit/test_dicom_processing.cpp
        tests/unit/test_autograd.cpp
        tests/unit/test_medical_models.cpp
        tests/unit/test_solvers.cpp
        tests/integration/test_medical_pipeline.cpp
        tests/integration/test_federated_learning.cpp
        tests/integration/test_clinical_workflow.cpp
        tests/validation/test_medical_accuracy.cpp
    )
    
    if(OPENALGEBRA_ENABLE_CUDA)
        list(APPEND TEST_SOURCES 
            tests/unit/test_cuda_medical_kernels.cpp
            tests/integration/test_gpu_medical_acceleration.cpp
        )
    endif()
    
    if(OPENALGEBRA_ENABLE_MPI)
        list(APPEND TEST_SOURCES 
            tests/unit/test_mpi_medical_operations.cpp
            tests/integration/test_distributed_medical_training.cpp
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
            OPENALGEBRA_MEDICAL_TEST_MODE
        )
        
        add_test(NAME ${test_name} COMMAND ${test_name})
        
        # Set test properties for medical validation
        if(test_name MATCHES "medical|clinical")
            set_tests_properties(${test_name} PROPERTIES
                TIMEOUT 300
                LABELS "medical;validation"
            )
        endif()
    endforeach()
    
    # Medical validation suite
    if(OPENALGEBRA_ENABLE_CLINICAL_VALIDATION)
        add_executable(clinical_validation_suite 
            tests/validation/medical_validation_suite.cpp
            tests/validation/clinical_test_cases.cpp
            tests/validation/regulatory_compliance_tests.cpp
        )
        
        target_link_libraries(clinical_validation_suite openalgebra)
        add_test(NAME clinical_validation_suite COMMAND clinical_validation_suite)
        set_tests_properties(clinical_validation_suite PROPERTIES
            TIMEOUT 600
            LABELS "clinical;validation;regulatory"
        )
    endif()
endif()

# Medical AI benchmarks
if(OPENALGEBRA_BUILD_BENCHMARKS)
    add_executable(medical_performance_suite 
        benchmarks/medical_performance_suite.cpp
        benchmarks/clinical_accuracy_benchmarks.cpp
        benchmarks/real_world_datasets.cpp
    )
    target_link_libraries(medical_performance_suite openalgebra)
    
    if(OPENALGEBRA_ENABLE_CUDA)
        add_executable(medical_cuda_benchmarks benchmarks/medical_cuda_benchmarks.cu)
        target_link_libraries(medical_cuda_benchmarks openalgebra)
        set_property(TARGET medical_cuda_benchmarks PROPERTY CUDA_RUNTIME_LIBRARY Static)
    endif()
    
    if(OPENALGEBRA_ENABLE_MPI)
        add_executable(distributed_medical_scaling benchmarks/distributed_medical_scaling.cpp)
        target_link_libraries(distributed_medical_scaling openalgebra)
    endif()
    
    # Medical AI model benchmarks
    add_executable(medical_model_benchmarks
        benchmarks/sparse_cnn_benchmarks.cpp
        benchmarks/medical_transformer_benchmarks.cpp
        benchmarks/federated_learning_benchmarks.cpp
    )
    target_link_libraries(medical_model_benchmarks openalgebra)
endif()

# Medical AI examples
if(OPENALGEBRA_BUILD_EXAMPLES)
    # Medical imaging examples
    add_executable(brain_tumor_segmentation examples/medical_imaging/brain_tumor_segmentation.cpp)
    target_link_libraries(brain_tumor_segmentation openalgebra)
    
    add_executable(chest_xray_classification examples/medical_imaging/chest_xray_classification.cpp)
    target_link_libraries(chest_xray_classification openalgebra)
    
    if(OPENALGEBRA_ENABLE_CUDA)
        add_executable(mri_reconstruction examples/medical_imaging/mri_reconstruction.cu)
        target_link_libraries(mri_reconstruction openalgebra)
        set_property(TARGET mri_reconstruction PROPERTY CUDA_RUNTIME_LIBRARY Static)
    endif()
    
    # Clinical NLP examples
    add_executable(clinical_notes_processing examples/clinical_nlp/clinical_notes_processing.cpp)
    target_link_libraries(clinical_notes_processing openalgebra)
    
    # Drug discovery examples
    add_executable(molecular_property_prediction examples/drug_discovery/molecular_property_prediction.cpp)
    target_link_libraries(molecular_property_prediction openalgebra)
    
    add_executable(protein_folding_simulation examples/drug_discovery/protein_folding_simulation.cpp)
    target_link_libraries(protein_folding_simulation openalgebra)
    
    # Federated learning example
    if(OPENALGEBRA_ENABLE_MPI)
        add_executable(federated_medical_training examples/federated_learning/federated_medical_training.cpp)
        target_link_libraries(federated_medical_training openalgebra)
    endif()
endif()

# Python bindings with medical AI support
if(OPENALGEBRA_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
    
    pybind11_add_module(openalgebra_python 
        src/python/openalgebra_bindings.cpp
        src/python/medical_ai_bindings.cpp
        src/python/sparse_matrix_bindings.cpp
        src/python/medical_tensor_bindings.cpp
        src/python/solver_bindings.cpp
    )
    
    target_link_libraries(openalgebra_python PRIVATE openalgebra)
    
    set_target_properties(openalgebra_python PROPERTIES
        OUTPUT_NAME openalgebra
        SUFFIX ${PYTHON_MODULE_EXTENSION}
    )
    
    # Python medical AI utilities
    install(FILES
        src/python/__init__.py
        src/python/sparse_matrix.py
        src/python/medical_models.py
        src/python/dicom_utils.py
        src/python/autograd.py
        src/python/clinical_validation.py
        DESTINATION ${Python_SITEARCH}/openalgebra
    )
endif()

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "OpenAlgebra Medical AI")
    set(DOXYGEN_PROJECT_BRIEF "High-Performance Sparse Linear Algebra for Medical AI")
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
    
    add_custom_target(medical_docs 
        COMMAND ${CMAKE_COMMAND} -E copy_directory 
                ${CMAKE_CURRENT_SOURCE_DIR}/docs 
                ${CMAKE_CURRENT_BINARY_DIR}/docs/medical
        DEPENDS docs
        COMMENT "Generating medical AI documentation"
    )
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "OpenAlgebra")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance Sparse Linear Algebra for Medical AI Model Development")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR "OpenAlgebra Medical AI Team")
set(CPACK_PACKAGE_CONTACT "contact@openalgebra-medical.org")
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
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "OpenAlgebra Medical AI Team")
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
message(STATUS "OpenAlgebra Medical AI Configuration Summary:")
message(STATUS "==========================================")
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
message(STATUS "Medical AI Features:")
message(STATUS "  Medical I/O support: ${OPENALGEBRA_ENABLE_MEDICAL_IO}")
message(STATUS "  Clinical validation: ${OPENALGEBRA_ENABLE_CLINICAL_VALIDATION}")
message(STATUS "  DCMTK (DICOM): ${DCMTK_FOUND}")
message(STATUS "  ITK (Medical imaging): ${ITK_FOUND}")
message(STATUS "  VTK (Visualization): ${VTK_FOUND}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Build tests: ${OPENALGEBRA_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${OPENALGEBRA_BUILD_BENCHMARKS}")
message(STATUS "  Build examples: ${OPENALGEBRA_BUILD_EXAMPLES}")
message(STATUS "  Build Python bindings: ${OPENALGEBRA_BUILD_PYTHON}")
message(STATUS "")
message(STATUS "Target Applications:")
message(STATUS "  - Medical image segmentation and analysis")
message(STATUS "  - Clinical decision support systems") 
message(STATUS "  - Drug discovery and molecular modeling")
message(STATUS "  - Federated learning for healthcare")
message(STATUS "  - Real-time medical image reconstruction")
message(STATUS "  - Multi-modal medical data fusion")
message(STATUS "==========================================")
message(STATUS "") 