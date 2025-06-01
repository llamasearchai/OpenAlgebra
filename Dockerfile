# OpenAlgebra Medical AI Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CMAKE_BUILD_TYPE=Release

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    # Python and pip
    python3 \
    python3-dev \
    python3-pip \
    # Medical imaging libraries
    dcmtk \
    libdcmtk-dev \
    libinsighttoolkit4-dev \
    libvtk9-dev \
    # Linear algebra libraries
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libsuitesparse-dev \
    # Sparse matrix libraries
    libsuitesparse-dev \
    libmetis-dev \
    libparmetis-dev \
    # Image processing
    libopencv-dev \
    libtiff5-dev \
    libpng-dev \
    libjpeg-dev \
    # Parallel computing
    libopenmpi-dev \
    openmpi-bin \
    # CUDA toolkit additions
    libcublas-dev-11-8 \
    libcusparse-dev-11-8 \
    libcusolver-dev-11-8 \
    libcurand-dev-11-8 \
    libcufft-dev-11-8 \
    # Additional utilities
    libeigen3-dev \
    libhdf5-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install modern CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh && \
    chmod +x cmake-3.27.0-linux-x86_64.sh && \
    ./cmake-3.27.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.27.0-linux-x86_64.sh

# Install Python dependencies for medical AI
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install additional medical Python packages
RUN pip3 install --no-cache-dir \
    pydicom==2.4.3 \
    nibabel==5.1.0 \
    SimpleITK==2.3.1 \
    radiomics==3.1.0 \
    scikit-image==0.21.0 \
    scipy==1.11.3 \
    numpy==1.24.3 \
    pandas==2.1.1 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.4.2 \
    aiofiles==23.2.1

# Create workspace
WORKDIR /workspace

# Copy source code
COPY . /workspace/

# Build OpenAlgebra
RUN mkdir -p build && cd build && \
    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DOPENALGEBRA_BUILD_TESTS=ON \
        -DOPENALGEBRA_BUILD_BENCHMARKS=ON \
        -DOPENALGEBRA_BUILD_EXAMPLES=ON \
        -DOPENALGEBRA_BUILD_PYTHON=ON \
        -DOPENALGEBRA_ENABLE_CUDA=ON \
        -DOPENALGEBRA_ENABLE_MPI=ON \
        -DOPENALGEBRA_ENABLE_OPENMP=ON \
        -DOPENALGEBRA_ENABLE_MEDICAL_IO=ON \
        -DOPENALGEBRA_ENABLE_CLINICAL_VALIDATION=ON \
        -DCUDA_ARCHITECTURES="70;75;80;86" && \
    ninja -j$(nproc) && \
    ninja install

# Run tests to validate build
RUN cd build && ctest --output-on-failure

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CUDA_VISIBLE_DEVICES=0

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime libraries
    libopenblas0 \
    libsuitesparse5 \
    libgomp1 \
    libquadmath0 \
    # Medical imaging runtime
    dcmtk \
    libinsighttoolkit4.13 \
    libvtk9.1 \
    # Python runtime
    python3 \
    python3-pip \
    # OpenMPI runtime
    libopenmpi3 \
    # CUDA runtime libraries (already included in base image)
    # Image processing runtime
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libtiff5 \
    libpng16-16 \
    libjpeg8 \
    # Additional utilities
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy built libraries and executables from builder
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/include /usr/local/include

# Install Python packages in production
RUN pip3 install --no-cache-dir \
    pydicom==2.4.3 \
    nibabel==5.1.0 \
    SimpleITK==2.3.1 \
    radiomics==3.1.0 \
    scikit-image==0.21.0 \
    scipy==1.11.3 \
    numpy==1.24.3 \
    pandas==2.1.1 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.4.2 \
    aiofiles==23.2.1

# Create non-root user for security
RUN useradd -m -s /bin/bash -u 1000 openalgebra && \
    mkdir -p /app /data /models /results && \
    chown -R openalgebra:openalgebra /app /data /models /results

# Copy application code
COPY --from=builder /workspace/src/api /app/api
COPY --from=builder /workspace/examples /app/examples
COPY --from=builder /workspace/python /app/python

# Set working directory
WORKDIR /app

# Switch to non-root user
USER openalgebra

# Create directories for medical data
RUN mkdir -p /app/data/dicom /app/data/nifti /app/models /app/results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Default command - start FastAPI medical AI service
CMD ["python3", "-m", "uvicorn", "api.medical_endpoints:app", "--host", "0.0.0.0", "--port", "8000"]

# Labels for metadata
LABEL maintainer="OpenAlgebra Medical AI Team <medical@openalgebra.org>"
LABEL version="1.0.0"
LABEL description="OpenAlgebra Medical AI - High-Performance Sparse Linear Algebra for Healthcare"
LABEL org.opencontainers.image.title="OpenAlgebra Medical AI"
LABEL org.opencontainers.image.description="Production-ready medical AI container with sparse linear algebra optimization"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="OpenAlgebra"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/llamasearchai/OpenAlgebra"
LABEL medical.compliance.hipaa="true"
LABEL medical.compliance.fda="preparing"
LABEL medical.modalities="CT,MRI,PET,X-ray,Ultrasound"
LABEL medical.applications="segmentation,classification,detection,registration"
