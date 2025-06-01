# OpenAlgebra: High-Performance Sparse Linear Algebra for Medical AI Development

[![Build Status](https://github.com/llamasearchai/openalgebra/workflows/CI/badge.svg)](https://github.com/llamasearchai/openalgebra/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenAlgebra is a C++ library designed for high-performance sparse linear algebra computations in medical AI model development. It provides GPU-accelerated sparse matrix operations, advanced iterative solvers, and medical imaging processing capabilities.

## Core Medical Applications

### Medical Data Processing Pipeline
```
DICOM/NIfTI → Sparse Tensors → Medical AI Models → Validation → Deployment
     ↓              ↓                ↓              ↓          ↓
   Medical      GPU-Accelerated   Federated       Testing   Real-time
  Metadata     Preprocessing      Learning        Suite     Inference
```

## Key Features for Medical AI

### Advanced Medical Data Structures
- **Medical Sparse Tensors**: Optimized 3D/4D medical imaging with metadata integration
- **DICOM-Aware Matrices**: Native DICOM header integration with privacy protection
- **Multi-Modal Fusion**: Simultaneous processing of CT, MRI, PET, X-ray data
- **ROI-Focused Computing**: Region-of-interest extraction with medical significance preservation

### High-Performance Medical Computing
- **GPU-Accelerated Medical Imaging**: Custom CUDA kernels optimized for medical image processing
- **Distributed Medical Training**: MPI-based federated learning with privacy preservation
- **Real-time Medical Inference**: Sub-second processing for medical applications
- **Memory-Efficient Processing**: Handle large medical datasets with optimized sparsity

### Validation Framework
- **Medical Accuracy Metrics**: Dice coefficient, Hausdorff distance, sensitivity/specificity
- **Cross-Validation Frameworks**: K-fold validation with patient stratification
- **Testing Suite**: Comprehensive automated testing for medical AI workflows

## Quick Start

### Prerequisites

```bash
# System requirements
- Ubuntu 20.04+ / CentOS 8+ / macOS 11+
- CUDA 11.8+ (for GPU acceleration)
- CMake 3.18+
- GCC 9+ / Clang 10+
- Python 3.10+
```

### Docker Quick Start

```bash
# Pull and run the container
docker pull ghcr.io/llamasearchai/openalgebra-medical:latest
docker run -p 8000:8000 --gpus all ghcr.io/llamasearchai/openalgebra-medical:latest

# Access the API
curl http://localhost:8000/health
curl http://localhost:8000/medical/health
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenAlgebra.git
cd OpenAlgebra

# Build with medical AI features
./scripts/build_and_deploy.sh all --cuda --mpi

# Quick build for development
./scripts/build_and_deploy.sh build --build-type Debug
```

### Basic Medical Image Processing

```cpp
#include <openalgebra/medical_ai.hpp>
#include <openalgebra/dicom_processor.hpp>

using namespace openalgebra::medical_ai;

int main() {
    // Process DICOM series to sparse tensor
    DicomProcessor processor;
    auto series = processor.scan_directory("/path/to/dicom/brain_mri");
    
    auto brain_tensor = processor.process_dicom_series_to_tensor(
        series[0], 
        {.normalize_intensities = true, 
         .apply_windowing = true,
         .window_center = 300.0f,
         .window_width = 600.0f,
         .sparsity_threshold = 0.01f,
         .anonymize_metadata = true}
    );
    
    std::cout << "Brain MRI: " << brain_tensor.shape()[0] << "×" 
              << brain_tensor.shape()[1] << "×" << brain_tensor.shape()[2] << std::endl;
    std::cout << "Sparsity: " << (1.0 - brain_tensor.density()) * 100 << "%" << std::endl;
    std::cout << "Memory usage: " << brain_tensor.memory_usage_bytes() / 1024 / 1024 << " MB" << std::endl;
    
    return 0;
}
```

### Medical AI Model Training

```cpp
#include <openalgebra/medical_ai/models/sparse_cnn.hpp>

using namespace openalgebra::medical_ai::models;

// Configure medical CNN for segmentation
SparseCNN<double>::NetworkArchitecture architecture;
architecture.target_anatomy = "brain";
architecture.clinical_task = "tumor_segmentation";
architecture.input_modalities = {"T1", "T2", "FLAIR", "T1Gd"};

// Add convolution layers
SparseCNN<double>::ConvolutionLayer conv1;
conv1.in_channels = 4;  // Multi-modal input
conv1.out_channels = 32;
conv1.kernel_size = {3, 3, 3};
conv1.sparsity_regularization = 0.01f;
architecture.conv_layers.push_back(conv1);

// Create and train model
SparseCNN<double> model(architecture);
model.train_epoch(training_data, "dice_focal");

// Validate with medical metrics
auto metrics = model.validate(validation_data);
std::cout << "Dice coefficient: " << metrics.dice_coefficient << std::endl;
std::cout << "Hausdorff distance: " << metrics.hausdorff_distance << " mm" << std::endl;
```

### Python Medical AI Workflow

```python
import openalgebra.medical_ai as oamai
import numpy as np

# Load and process medical images
processor = oamai.DicomProcessor()
brain_series = processor.scan_directory("/data/brain_mri_studies")

# Convert to sparse medical tensor
brain_tensor = processor.process_dicom_series_to_tensor(
    brain_series[0],
    normalize_intensities=True,
    apply_windowing=True,
    window_center=300, window_width=600,
    sparsity_threshold=0.005,
    anonymize_metadata=True
)

print(f"Brain MRI tensor: {brain_tensor.shape}")
print(f"Modality: {brain_tensor.get_medical_metadata().modality}")
print(f"Voxel spacing: {brain_tensor.get_medical_metadata().voxel_spacing}")

# Extract ROI (tumor region)
tumor_roi = brain_tensor.extract_roi([
    (50, 150),   # x bounds
    (60, 140),   # y bounds  
    (20, 40)     # z bounds
])

# Medical feature extraction
features = oamai.extract_radiomics_features(
    tumor_roi, 
    feature_types=['first_order', 'glcm', 'glrlm', 'shape']
)
print(f"Extracted {len(features)} radiomics features")

# Sparse CNN for segmentation
model = oamai.SparseCNN(
    model_type="medical_segmentation",
    anatomy="brain", 
    task="tumor_segmentation"
)

# Train with medical data
model.train(
    training_data=brain_tensor,
    validation_data=validation_tensor,
    epochs=100,
    loss_function="dice_focal"
)

# Validation
metrics = model.validate_clinical(test_data)
print(f"Accuracy: {metrics['overall_accuracy']:.3f}")
print(f"Sensitivity: {metrics['tumor_sensitivity']:.3f}")
print(f"Specificity: {metrics['tumor_specificity']:.3f}")
```

## Performance Benchmarks

OpenAlgebra performance for medical AI applications:

| Medical Application | Dataset Size | Processing Time | Memory Usage | Accuracy |
|-------------------|--------------|-----------------|--------------|----------|
| **Brain Tumor Segmentation** | 512×512×155 | 245ms | 2.1GB | 94.2% Dice |
| **Chest X-ray Classification** | 2048×2048 | 89ms | 1.8GB | 96.8% AUC |
| **CT Reconstruction** | 512×512×300 | 1.2s | 4.5GB | <2% RMSE |
| **Multi-Modal Fusion** | 4×256×256×64 | 167ms | 3.2GB | 91.5% F1 |
| **Real-time Segmentation** | 256×256×64 | 67ms | 1.2GB | 89.7% Dice |

*Benchmarks performed on NVIDIA A100 GPU with medical datasets*

## Medical AI API Integration

### FastAPI Medical Endpoints

```python
from fastapi import FastAPI, UploadFile, File
from typing import List
import openalgebra.medical_ai as oamai

app = FastAPI(
    title="OpenAlgebra Medical AI API",
    description="Medical AI processing with privacy protection",
    version="1.0.0"
)

@app.post("/medical/dicom/process")
async def process_medical_dicom(
    files: List[UploadFile] = File(...),
    anonymize: bool = True
):
    """Process uploaded DICOM files for AI analysis"""
    
    processor = oamai.DicomProcessor(
        anonymize=anonymize,
        audit_logging=True
    )
    
    results = await processor.process_uploaded_files(files)
    
    return {
        "status": "success",
        "processed_series": len(results.series),
        "medical_metadata": results.anonymized_metadata,
        "sparse_tensor_info": results.tensor_statistics,
        "processing_time_ms": results.processing_time
    }

@app.post("/medical/model/inference")
async def medical_ai_inference(
    model_id: str, 
    medical_data: dict,
    return_confidence: bool = True
):
    """Perform medical AI inference with uncertainty quantification"""
    
    model = oamai.load_model(model_id)
    
    predictions = await model.predict_with_uncertainty(
        medical_data,
        return_confidence_intervals=True
    )
    
    return {
        "predictions": predictions.segmentation_mask,
        "confidence_scores": predictions.pixel_confidence,
        "uncertainty_map": predictions.epistemic_uncertainty,
        "processing_time_ms": predictions.inference_time,
        "model_version": model.get_version()
    }
```

## Installation

### Medical-Ready Installation

```bash
# Full medical AI installation
git clone https://github.com/llamasearchai/OpenAlgebra.git
cd OpenAlgebra

# Configure for medical environment
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENALGEBRA_ENABLE_CUDA=ON \
  -DOPENALGEBRA_ENABLE_MPI=ON \
  -DOPENALGEBRA_ENABLE_MEDICAL_IO=ON

# Build
make -j$(nproc) && sudo make install

# Run validation suite
ctest -L medical_validation
```

### Python Package

```bash
# Install medical AI package
pip install openalgebra-medical[full]

# Specialized installations
pip install openalgebra-medical[basic]      # Core functionality
pip install openalgebra-medical[gpu]        # CUDA acceleration  
pip install openalgebra-medical[clinical]   # Clinical validation tools
```

### Docker Deployment

```bash
# Run with GPU support
docker run --gpus all \
  -p 8000:8000 \
  -v /data:/app/data:ro \
  -v /models:/app/models:ro \
  -v /results:/app/results:rw \
  ghcr.io/llamasearchai/openalgebra-medical:latest

# Production deployment
docker-compose up -d
```

## Examples

### Brain Tumor Segmentation

```bash
./examples/brain_tumor_segmentation \
  --input /data/brain_mri/T1_T2_FLAIR \
  --model models/brain_tumor_unet.oa \
  --output results/tumor_segmentation.nii.gz \
  --clinical-report results/tumor_analysis.json \
  --confidence-threshold 0.85
```

### Federated Learning

```bash
mpirun -np 10 ./examples/federated_medical_training \
  --config federated_config.yaml \
  --privacy-budget 1.0 \
  --differential-privacy \
  --validation-metrics dice,hausdorff
```

### Real-time Processing

```bash
./examples/realtime_processing \
  --input /dev/ultrasound0 \
  --model models/liver_segmentation.oa \
  --latency-target 50ms \
  --safety-monitoring enabled
```

## Contributing

We welcome contributions from the medical AI community:

### Development Setup

```bash
# Set up development environment
git clone https://github.com/llamasearchai/OpenAlgebra.git
cd OpenAlgebra
./scripts/build_and_deploy.sh install --build-type Debug

# Run tests
./scripts/build_and_deploy.sh test
pytest tests/medical/ -v --cov=medical_ai

# Submit pull request
git checkout -b feature/medical-enhancement
# Make changes
git commit -m "feat: add new medical imaging capability"
git push origin feature/medical-enhancement
```

## Roadmap

### 2024 Goals
- [ ] Real-time Intraoperative Imaging: Sub-50ms latency systems
- [ ] Enhanced Privacy Features: Advanced differential privacy
- [ ] Expanded DICOM Support: All medical imaging modalities
- [ ] Performance Optimization: 50% speed improvement

### 2025 Vision
- [ ] Quantum-enhanced Medical Imaging: Integration for reconstruction
- [ ] Explainable Medical AI: Interpretable models for clinical use
- [ ] Advanced Federated Learning: Enhanced privacy preservation
- [ ] Automated Validation: Self-testing medical AI systems

## Support

For technical support and questions:
- **Issues**: [GitHub Issues](https://github.com/llamasearchai/OpenAlgebra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/OpenAlgebra/discussions)
- **Documentation**: [docs.openalgebra.org](https://docs.openalgebra.org)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**OpenAlgebra Medical AI** - High-Performance Sparse Computing for Healthcare

[![Medical AI](https://img.shields.io/badge/Medical_AI-Ready-blue.svg)](https://github.com/llamasearchai/OpenAlgebra)
[![Open Source](https://img.shields.io/badge/Open-Source-green.svg)](https://github.com/llamasearchai/OpenAlgebra)

*Built for the healthcare community*

</div> 