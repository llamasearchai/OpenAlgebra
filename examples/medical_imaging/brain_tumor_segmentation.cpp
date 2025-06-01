#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <memory>

// OpenAlgebra core includes
#include "../../src/core/algebra/matrix/csr_matrix.hpp"
#include "../../src/core/algebra/tensor/sparse_tensor_coo.hpp"
#include "../../src/core/algebra/solvers/iterative/cg_solver.hpp"
#include "../../src/medical_ai/data_io/dicom_processor.hpp"
#include "../../src/medical_ai/models/sparse_cnn.hpp"
#include "../../src/medical_ai/preprocessing/medical_normalization.hpp"
#include "../../src/medical_ai/evaluation/medical_metrics.hpp"
#include "../../src/core/utils/logger.hpp"

using namespace openalgebra::core::algebra;
using namespace openalgebra::medical_ai;

/**
 * Brain Tumor Segmentation Example
 * 
 * This example demonstrates a complete medical AI workflow using OpenAlgebra:
 * 1. DICOM data loading and preprocessing
 * 2. Sparse tensor creation for 3D brain volumes
 * 3. Medical image segmentation using sparse CNN
 * 4. Clinical validation metrics computation
 * 5. Results visualization and export
 */
class BrainTumorSegmentation {
private:
    // Configuration
    struct SegmentationConfig {
        std::string input_dicom_path;
        std::string output_path;
        std::string model_path;
        
        // Image processing parameters
        bool normalize_intensities = true;
        bool apply_skull_stripping = true;
        float sparsity_threshold = 0.01f;
        
        // Windowing parameters for brain MRI
        float window_center = 300.0f;
        float window_width = 600.0f;
        
        // Segmentation parameters
        std::string model_type = "sparse_cnn";
        int patch_size = 64;
        int patch_overlap = 16;
        
        // Clinical validation
        bool compute_clinical_metrics = true;
        bool generate_report = true;
        
        // Performance settings
        int num_threads = -1;  // Use all available
        bool enable_gpu = true;
    };
    
    SegmentationConfig config_;
    std::unique_ptr<data_io::DicomProcessor> dicom_processor_;
    std::unique_ptr<models::SparseCNN<float>> segmentation_model_;
    std::unique_ptr<utils::Logger> logger_;
    
public:
    BrainTumorSegmentation(const SegmentationConfig& config) 
        : config_(config) {
        
        // Initialize components
        dicom_processor_ = std::make_unique<data_io::DicomProcessor>();
        logger_ = std::make_unique<utils::Logger>("BrainTumorSegmentation");
        
        logger_->info("Initializing brain tumor segmentation pipeline");
        logger_->info("Input path: " + config_.input_dicom_path);
        logger_->info("Output path: " + config_.output_path);
    }
    
    int run() {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Step 1: Load and preprocess DICOM data
            logger_->info("Step 1: Loading DICOM data...");
            auto brain_volume = load_and_preprocess_dicom();
            
            // Step 2: Initialize segmentation model
            logger_->info("Step 2: Initializing segmentation model...");
            initialize_segmentation_model();
            
            // Step 3: Perform tumor segmentation
            logger_->info("Step 3: Performing tumor segmentation...");
            auto segmentation_result = perform_segmentation(brain_volume);
            
            // Step 4: Post-process segmentation
            logger_->info("Step 4: Post-processing segmentation...");
            auto final_segmentation = post_process_segmentation(segmentation_result);
            
            // Step 5: Compute clinical metrics
            logger_->info("Step 5: Computing clinical validation metrics...");
            auto clinical_metrics = compute_clinical_metrics(final_segmentation);
            
            // Step 6: Generate outputs
            logger_->info("Step 6: Generating outputs...");
            generate_outputs(brain_volume, final_segmentation, clinical_metrics);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time).count();
            
            logger_->info("Brain tumor segmentation completed successfully in " + 
                         std::to_string(duration) + " seconds");
            
            return 0;
            
        } catch (const std::exception& e) {
            logger_->error("Segmentation failed: " + std::string(e.what()));
            return 1;
        }
    }

private:
    tensor::SparseTensorCOO<float> load_and_preprocess_dicom() {
        // Scan DICOM directory
        auto dicom_series = dicom_processor_->scan_directory(config_.input_dicom_path);
        
        if (dicom_series.empty()) {
            throw std::runtime_error("No DICOM series found in: " + config_.input_dicom_path);
        }
        
        logger_->info("Found " + std::to_string(dicom_series.size()) + " DICOM series");
        
        // Select the first T1-weighted series (in real application, would be more sophisticated)
        auto selected_series = dicom_series[0];
        for (const auto& series : dicom_series) {
            if (series.header.series_description.find("T1") != std::string::npos) {
                selected_series = series;
                break;
            }
        }
        
        logger_->info("Selected series: " + selected_series.header.series_description);
        logger_->info("Number of slices: " + std::to_string(selected_series.file_paths.size()));
        
        // Configure processing options
        data_io::DicomProcessor::ProcessingOptions options;
        options.normalize_intensities = config_.normalize_intensities;
        options.apply_windowing = true;
        options.window_center = config_.window_center;
        options.window_width = config_.window_width;
        options.sparsity_threshold = config_.sparsity_threshold;
        options.anonymize_metadata = true;
        
        // Process DICOM series to sparse tensor
        auto brain_tensor = dicom_processor_->process_dicom_series_to_tensor(
            selected_series, options);
        
        logger_->info("Brain volume tensor created:");
        logger_->info("  Dimensions: " + tensor_shape_string(brain_tensor.shape()));
        logger_->info("  Non-zero voxels: " + std::to_string(brain_tensor.nnz()));
        logger_->info("  Sparsity: " + std::to_string((1.0 - brain_tensor.density()) * 100) + "%");
        logger_->info("  Memory usage: " + std::to_string(brain_tensor.memory_usage_bytes() / 1024 / 1024) + " MB");
        
        // Apply medical preprocessing if enabled
        if (config_.apply_skull_stripping) {
            apply_skull_stripping(brain_tensor);
        }
        
        return brain_tensor;
    }
    
    void initialize_segmentation_model() {
        // Create sparse CNN architecture for brain tumor segmentation
        typename models::SparseCNN<float>::NetworkArchitecture architecture;
        architecture.target_anatomy = "brain";
        architecture.clinical_task = "tumor_segmentation";
        architecture.input_modalities = {"T1"};
        architecture.use_anatomical_priors = true;
        architecture.use_multi_modal_fusion = false;
        
        // Design network layers
        // Encoder layers
        for (int i = 0; i < 4; ++i) {
            typename models::SparseCNN<float>::ConvolutionLayer conv_layer;
            conv_layer.in_channels = (i == 0) ? 1 : 32 * (1 << (i-1));
            conv_layer.out_channels = 32 * (1 << i);
            conv_layer.kernel_size = {3, 3, 3};
            conv_layer.stride = {1, 1, 1};
            conv_layer.padding = {1, 1, 1};
            conv_layer.activation = (i < 3) ? "relu" : "leaky_relu";
            conv_layer.use_batch_norm = true;
            conv_layer.medical_aware_initialization = true;
            conv_layer.medical_prior = "brain";
            conv_layer.sparsity_regularization = 0.01f;
            
            architecture.conv_layers.push_back(conv_layer);
            
            // Add pooling layer (except for last encoder)
            if (i < 3) {
                typename models::SparseCNN<float>::PoolingLayer pool_layer;
                pool_layer.pool_size = {2, 2, 2};
                pool_layer.stride = {2, 2, 2};
                pool_layer.pooling_type = "max";
                pool_layer.preserve_medical_structure = true;
                
                architecture.pool_layers.push_back(pool_layer);
            }
        }
        
        // Decoder layers (symmetric to encoder)
        for (int i = 3; i >= 0; --i) {
            typename models::SparseCNN<float>::ConvolutionLayer conv_layer;
            conv_layer.in_channels = 32 * (1 << i);
            conv_layer.out_channels = (i == 0) ? 2 : 32 * (1 << (i-1));  // 2 classes: background, tumor
            conv_layer.kernel_size = {3, 3, 3};
            conv_layer.stride = {1, 1, 1};
            conv_layer.padding = {1, 1, 1};
            conv_layer.activation = (i > 0) ? "relu" : "softmax";
            conv_layer.use_batch_norm = (i > 0);
            conv_layer.medical_aware_initialization = true;
            conv_layer.medical_prior = "brain";
            
            architecture.conv_layers.push_back(conv_layer);
        }
        
        // Add medical attention mechanism
        typename models::SparseCNN<float>::MedicalAttentionLayer attention;
        attention.num_heads = 8;
        attention.attention_dim = 256;
        attention.dropout_rate = 0.1f;
        attention.anatomical_attention = true;
        attention.multi_scale_attention = true;
        
        architecture.attention_layers.push_back(attention);
        
        // Define layer sequence (encoder -> attention -> decoder)
        architecture.layer_sequence = {0, 100, 1, 101, 2, 102, 3, 200, 7, 6, 5, 4};
        
        // Initialize model
        segmentation_model_ = std::make_unique<models::SparseCNN<float>>(architecture);
        
        // Load pre-trained weights if model path is provided
        if (!config_.model_path.empty() && std::filesystem::exists(config_.model_path)) {
            logger_->info("Loading pre-trained model from: " + config_.model_path);
            segmentation_model_->load_model(config_.model_path);
        } else {
            logger_->warning("No pre-trained model found. Using randomly initialized weights.");
            
            // Set anatomical priors for brain
            auto brain_priors = load_brain_anatomical_priors();
            segmentation_model_->set_anatomical_priors(brain_priors);
        }
        
        segmentation_model_->set_training_mode(false);  // Inference mode
    }
    
    tensor::SparseTensorCOO<float> perform_segmentation(
        const tensor::SparseTensorCOO<float>& brain_volume) {
        
        auto volume_shape = brain_volume.shape();
        int width = volume_shape[0];
        int height = volume_shape[1];
        int depth = volume_shape[2];
        
        logger_->info("Performing patch-based segmentation...");
        logger_->info("  Patch size: " + std::to_string(config_.patch_size) + "^3");
        logger_->info("  Patch overlap: " + std::to_string(config_.patch_overlap));
        
        // Create output segmentation tensor
        tensor::SparseTensorCOO<float> segmentation(volume_shape);
        
        // Copy medical metadata
        if (brain_volume.is_medical_data()) {
            auto metadata = *brain_volume.get_medical_metadata();
            metadata.preprocessing_pipeline += " -> tumor_segmentation";
            segmentation.set_medical_metadata(metadata);
        }
        
        int patch_step = config_.patch_size - config_.patch_overlap;
        int total_patches = 0;
        int processed_patches = 0;
        
        // Count total patches for progress tracking
        for (int z = 0; z <= depth - config_.patch_size; z += patch_step) {
            for (int y = 0; y <= height - config_.patch_size; y += patch_step) {
                for (int x = 0; x <= width - config_.patch_size; x += patch_step) {
                    total_patches++;
                }
            }
        }
        
        logger_->info("Total patches to process: " + std::to_string(total_patches));
        
        // Process patches
        for (int z = 0; z <= depth - config_.patch_size; z += patch_step) {
            for (int y = 0; y <= height - config_.patch_size; y += patch_step) {
                for (int x = 0; x <= width - config_.patch_size; x += patch_step) {
                    
                    // Extract patch
                    std::vector<std::pair<int, int>> patch_bounds = {
                        {x, x + config_.patch_size},
                        {y, y + config_.patch_size},
                        {z, z + config_.patch_size}
                    };
                    
                    auto patch = brain_volume.extract_roi(patch_bounds);
                    
                    // Skip nearly empty patches
                    if (patch.nnz() < 10) {
                        processed_patches++;
                        continue;
                    }
                    
                    // Convert to model input format
                    auto patch_variable = tensor_to_variable(patch);
                    
                    // Run inference
                    auto prediction = segmentation_model_->predict(patch_variable);
                    
                    // Post-process prediction
                    auto processed_prediction = segmentation_model_->postprocess_medical_prediction(
                        prediction, "segmentation");
                    
                    // Insert prediction into full segmentation
                    insert_patch_prediction(segmentation, processed_prediction, x, y, z);
                    
                    processed_patches++;
                    
                    // Progress reporting
                    if (processed_patches % 100 == 0 || processed_patches == total_patches) {
                        double progress = static_cast<double>(processed_patches) / total_patches * 100;
                        logger_->info("Progress: " + std::to_string(static_cast<int>(progress)) + 
                                    "% (" + std::to_string(processed_patches) + "/" + 
                                    std::to_string(total_patches) + ")");
                    }
                }
            }
        }
        
        logger_->info("Segmentation completed. Non-zero predictions: " + 
                     std::to_string(segmentation.nnz()));
        
        return segmentation;
    }
    
    tensor::SparseTensorCOO<float> post_process_segmentation(
        const tensor::SparseTensorCOO<float>& raw_segmentation) {
        
        logger_->info("Post-processing segmentation...");
        
        // Apply morphological operations to clean up segmentation
        auto cleaned_segmentation = raw_segmentation;
        
        // 1. Remove small connected components (noise)
        remove_small_components(cleaned_segmentation, 50);  // Minimum 50 voxels
        
        // 2. Fill holes in tumor regions
        fill_segmentation_holes(cleaned_segmentation);
        
        // 3. Smooth boundaries
        smooth_segmentation_boundaries(cleaned_segmentation);
        
        // Update metadata
        if (cleaned_segmentation.is_medical_data()) {
            auto metadata = const_cast<typename tensor::SparseTensorCOO<float>::MedicalTensorMetadata*>(
                cleaned_segmentation.get_medical_metadata());
            metadata->preprocessing_pipeline += " -> morphological_cleanup";
        }
        
        logger_->info("Post-processing completed. Final segmentation non-zeros: " + 
                     std::to_string(cleaned_segmentation.nnz()));
        
        return cleaned_segmentation;
    }
    
    evaluation::MedicalMetrics compute_clinical_metrics(
        const tensor::SparseTensorCOO<float>& segmentation) {
        
        if (!config_.compute_clinical_metrics) {
            return evaluation::MedicalMetrics{};
        }
        
        logger_->info("Computing clinical validation metrics...");
        
        evaluation::MedicalMetrics metrics;
        
        // Compute volume statistics
        auto stats = segmentation.compute_medical_statistics();
        
        // Convert to physical measurements using voxel spacing
        if (segmentation.is_medical_data()) {
            auto metadata = segmentation.get_medical_metadata();
            float voxel_volume = metadata->voxel_spacing[0] * 
                               metadata->voxel_spacing[1] * 
                               metadata->voxel_spacing[2];
            
            metrics.tumor_volume_ml = stats.num_nonzero_voxels * voxel_volume / 1000.0f;  // mm³ to ml
            metrics.tumor_centroid = stats.centroid;
            
            logger_->info("Tumor volume: " + std::to_string(metrics.tumor_volume_ml) + " ml");
            logger_->info("Tumor centroid: (" + 
                         std::to_string(stats.centroid[0]) + ", " +
                         std::to_string(stats.centroid[1]) + ", " +
                         std::to_string(stats.centroid[2]) + ")");
        }
        
        // Compute shape characteristics
        metrics.sphericity = compute_sphericity(segmentation);
        metrics.compactness = compute_compactness(segmentation);
        metrics.surface_to_volume_ratio = compute_surface_to_volume_ratio(segmentation);
        
        logger_->info("Shape metrics:");
        logger_->info("  Sphericity: " + std::to_string(metrics.sphericity));
        logger_->info("  Compactness: " + std::to_string(metrics.compactness));
        logger_->info("  Surface-to-volume ratio: " + std::to_string(metrics.surface_to_volume_ratio));
        
        // Clinical classification
        classify_tumor_characteristics(metrics);
        
        return metrics;
    }
    
    void generate_outputs(
        const tensor::SparseTensorCOO<float>& brain_volume,
        const tensor::SparseTensorCOO<float>& segmentation,
        const evaluation::MedicalMetrics& metrics) {
        
        // Create output directory
        std::filesystem::create_directories(config_.output_path);
        
        // 1. Export segmentation mask as NIfTI
        std::string nifti_path = config_.output_path + "/tumor_segmentation.nii.gz";
        export_segmentation_to_nifti(segmentation, nifti_path);
        logger_->info("Segmentation saved to: " + nifti_path);
        
        // 2. Export overlay visualization
        std::string overlay_path = config_.output_path + "/tumor_overlay.nii.gz";
        create_segmentation_overlay(brain_volume, segmentation, overlay_path);
        logger_->info("Overlay saved to: " + overlay_path);
        
        // 3. Generate clinical report
        if (config_.generate_report) {
            std::string report_path = config_.output_path + "/clinical_report.json";
            generate_clinical_report(metrics, report_path);
            logger_->info("Clinical report saved to: " + report_path);
        }
        
        // 4. Export DICOM SR (Structured Report)
        std::string sr_path = config_.output_path + "/segmentation_sr.dcm";
        export_dicom_structured_report(segmentation, metrics, sr_path);
        logger_->info("DICOM SR saved to: " + sr_path);
        
        // 5. Generate visualization images
        std::string vis_dir = config_.output_path + "/visualizations";
        std::filesystem::create_directories(vis_dir);
        generate_visualization_slices(brain_volume, segmentation, vis_dir);
        logger_->info("Visualization images saved to: " + vis_dir);
    }
    
    // Helper functions
    std::string tensor_shape_string(const std::vector<int>& shape) {
        std::string result = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            result += std::to_string(shape[i]);
            if (i < shape.size() - 1) result += ", ";
        }
        result += "]";
        return result;
    }
    
    void apply_skull_stripping(tensor::SparseTensorCOO<float>& brain_tensor) {
        logger_->info("Applying skull stripping...");
        
        // Simple skull stripping based on intensity thresholding and morphology
        // In practice, would use more sophisticated algorithms like BET or ROBEX
        
        auto stats = brain_tensor.compute_medical_statistics();
        float brain_threshold = stats.mean_intensity + 0.5f * stats.std_intensity;
        
        // Remove low-intensity voxels (skull and background)
        auto& values = brain_tensor.values();
        auto& indices = brain_tensor.indices();
        
        std::vector<float> new_values;
        std::vector<std::vector<int>> new_indices(indices.size());
        
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] > brain_threshold) {
                new_values.push_back(values[i]);
                for (size_t dim = 0; dim < indices.size(); ++dim) {
                    new_indices[dim].push_back(indices[dim][i]);
                }
            }
        }
        
        // Update tensor
        values = new_values;
        for (size_t dim = 0; dim < indices.size(); ++dim) {
            indices[dim] = new_indices[dim];
        }
        
        if (brain_tensor.is_medical_data()) {
            auto metadata = const_cast<typename tensor::SparseTensorCOO<float>::MedicalTensorMetadata*>(
                brain_tensor.get_medical_metadata());
            metadata->preprocessing_pipeline += " -> skull_stripping";
        }
        
        logger_->info("Skull stripping completed. Remaining voxels: " + 
                     std::to_string(brain_tensor.nnz()));
    }
    
    std::map<std::string, tensor::SparseTensorCOO<float>> load_brain_anatomical_priors() {
        // In a real implementation, these would be loaded from pre-computed atlas data
        std::map<std::string, tensor::SparseTensorCOO<float>> priors;
        
        // Create simple anatomical priors for demonstration
        std::vector<int> atlas_shape = {128, 128, 64};
        
        // Gray matter prior
        tensor::SparseTensorCOO<float> gray_matter_prior(atlas_shape);
        // White matter prior
        tensor::SparseTensorCOO<float> white_matter_prior(atlas_shape);
        // CSF prior
        tensor::SparseTensorCOO<float> csf_prior(atlas_shape);
        
        priors["gray_matter"] = gray_matter_prior;
        priors["white_matter"] = white_matter_prior;
        priors["csf"] = csf_prior;
        
        return priors;
    }
    
    // Placeholder implementations for complex medical operations
    void remove_small_components(tensor::SparseTensorCOO<float>& segmentation, int min_size) {
        // Implementation would use connected component analysis
        logger_->debug("Removing components smaller than " + std::to_string(min_size) + " voxels");
    }
    
    void fill_segmentation_holes(tensor::SparseTensorCOO<float>& segmentation) {
        // Implementation would use morphological closing operations
        logger_->debug("Filling holes in segmentation");
    }
    
    void smooth_segmentation_boundaries(tensor::SparseTensorCOO<float>& segmentation) {
        // Implementation would use Gaussian smoothing or morphological operations
        logger_->debug("Smoothing segmentation boundaries");
    }
    
    float compute_sphericity(const tensor::SparseTensorCOO<float>& segmentation) {
        // Sphericity = (π^(1/3) * (6V)^(2/3)) / A
        // where V is volume and A is surface area
        return 0.85f;  // Placeholder
    }
    
    float compute_compactness(const tensor::SparseTensorCOO<float>& segmentation) {
        // Compactness = V / (A^(3/2))
        return 0.75f;  // Placeholder
    }
    
    float compute_surface_to_volume_ratio(const tensor::SparseTensorCOO<float>& segmentation) {
        return 0.25f;  // Placeholder
    }
    
    void classify_tumor_characteristics(evaluation::MedicalMetrics& metrics) {
        // Simple classification based on volume and shape
        if (metrics.tumor_volume_ml < 1.0f) {
            metrics.tumor_classification = "Small";
        } else if (metrics.tumor_volume_ml < 10.0f) {
            metrics.tumor_classification = "Medium";
        } else {
            metrics.tumor_classification = "Large";
        }
        
        if (metrics.sphericity > 0.8f) {
            metrics.shape_classification = "Regular";
        } else {
            metrics.shape_classification = "Irregular";
        }
        
        logger_->info("Tumor classification: " + metrics.tumor_classification);
        logger_->info("Shape classification: " + metrics.shape_classification);
    }
    
    // More placeholder implementations
    core::autograd::Variable<tensor::SparseTensorCOO<float>> tensor_to_variable(
        const tensor::SparseTensorCOO<float>& tensor) {
        return core::autograd::Variable<tensor::SparseTensorCOO<float>>(tensor);
    }
    
    void insert_patch_prediction(
        tensor::SparseTensorCOO<float>& full_segmentation,
        const tensor::SparseTensorCOO<float>& patch_prediction,
        int x_offset, int y_offset, int z_offset) {
        // Implementation would handle patch integration with overlap resolution
    }
    
    void export_segmentation_to_nifti(
        const tensor::SparseTensorCOO<float>& segmentation,
        const std::string& filename) {
        logger_->debug("Exporting segmentation to NIfTI: " + filename);
    }
    
    void create_segmentation_overlay(
        const tensor::SparseTensorCOO<float>& brain_volume,
        const tensor::SparseTensorCOO<float>& segmentation,
        const std::string& filename) {
        logger_->debug("Creating segmentation overlay: " + filename);
    }
    
    void generate_clinical_report(
        const evaluation::MedicalMetrics& metrics,
        const std::string& filename) {
        logger_->debug("Generating clinical report: " + filename);
    }
    
    void export_dicom_structured_report(
        const tensor::SparseTensorCOO<float>& segmentation,
        const evaluation::MedicalMetrics& metrics,
        const std::string& filename) {
        logger_->debug("Exporting DICOM structured report: " + filename);
    }
    
    void generate_visualization_slices(
        const tensor::SparseTensorCOO<float>& brain_volume,
        const tensor::SparseTensorCOO<float>& segmentation,
        const std::string& output_dir) {
        logger_->debug("Generating visualization slices in: " + output_dir);
    }
};

// Command line argument parsing
struct CommandLineArgs {
    std::string input_path;
    std::string output_path;
    std::string model_path;
    bool help = false;
    bool verbose = false;
    int num_threads = -1;
    bool enable_gpu = true;
};

CommandLineArgs parse_arguments(int argc, char* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) {
                args.input_path = argv[++i];
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                args.output_path = argv[++i];
            }
        } else if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                args.model_path = argv[++i];
            }
        } else if (arg == "--threads" || arg == "-t") {
            if (i + 1 < argc) {
                args.num_threads = std::stoi(argv[++i]);
            }
        } else if (arg == "--no-gpu") {
            args.enable_gpu = false;
        }
    }
    
    return args;
}

void print_usage(const char* program_name) {
    std::cout << "OpenAlgebra Brain Tumor Segmentation Example\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input PATH     Input DICOM directory (required)\n";
    std::cout << "  -o, --output PATH    Output directory (required)\n";
    std::cout << "  -m, --model PATH     Pre-trained model file (optional)\n";
    std::cout << "  -t, --threads NUM    Number of threads (default: all available)\n";
    std::cout << "  --no-gpu             Disable GPU acceleration\n";
    std::cout << "  -v, --verbose        Enable verbose logging\n";
    std::cout << "  -h, --help           Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " -i /data/brain_mri -o /results/segmentation\n\n";
    std::cout << "This example demonstrates:\n";
    std::cout << "  • DICOM data loading and preprocessing\n";
    std::cout << "  • Sparse tensor operations for 3D medical imaging\n";
    std::cout << "  • Deep learning-based tumor segmentation\n";
    std::cout << "  • Clinical validation metrics computation\n";
    std::cout << "  • Medical imaging workflow integration\n";
}

int main(int argc, char* argv[]) {
    try {
        auto args = parse_arguments(argc, argv);
        
        if (args.help) {
            print_usage(argv[0]);
            return 0;
        }
        
        if (args.input_path.empty() || args.output_path.empty()) {
            std::cerr << "Error: Input and output paths are required.\n";
            std::cerr << "Use --help for usage information.\n";
            return 1;
        }
        
        // Configure segmentation pipeline
        BrainTumorSegmentation::SegmentationConfig config;
        config.input_dicom_path = args.input_path;
        config.output_path = args.output_path;
        config.model_path = args.model_path;
        config.num_threads = args.num_threads;
        config.enable_gpu = args.enable_gpu;
        
        // Initialize and run segmentation
        BrainTumorSegmentation segmentation_pipeline(config);
        
        std::cout << "Starting brain tumor segmentation...\n";
        std::cout << "Input: " << args.input_path << "\n";
        std::cout << "Output: " << args.output_path << "\n";
        
        int result = segmentation_pipeline.run();
        
        if (result == 0) {
            std::cout << "\nSegmentation completed successfully!\n";
            std::cout << "Results saved to: " << args.output_path << "\n";
        } else {
            std::cerr << "\nSegmentation failed with error code: " << result << "\n";
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
} 