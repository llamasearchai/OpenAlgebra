//! Medical Types and Data Structures
//!
//! Common medical data structures and types used across all medical AI modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::sparse::SparseMatrix;
use anyhow::Result;

// Radiology specific types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagingFinding {
    pub finding_id: String,
    pub finding_type: String,
    pub location: AnatomicalLocation,
    pub description: String,
    pub severity: SeverityLevel,
    pub confidence: f64,
    pub measurements: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnatomicalLocation {
    pub organ: String,
    pub region: String,
    pub coordinates: Option<Coordinates3D>,
    pub laterality: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coordinates3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    Minimal,
    Mild,
    Moderate,
    Severe,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionParameters {
    pub slice_thickness: f64,
    pub pixel_spacing: (f64, f64),
    pub matrix_size: (u32, u32),
    pub field_of_view: (f64, f64),
    pub contrast_agent: Option<String>,
    pub scan_parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    pub modality: String,
    pub body_part: String,
    pub patient_position: String,
    pub acquisition_date: DateTime<Utc>,
    pub manufacturer: String,
    pub model_name: String,
    pub acquisition_parameters: AcquisitionParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessmentModule {
    pub assessment_algorithms: Vec<String>,
    pub quality_thresholds: HashMap<String, f64>,
}

impl QualityAssessmentModule {
    pub fn new() -> Self {
        Self {
            assessment_algorithms: vec![
                "noise_analysis".to_string(),
                "contrast_assessment".to_string(),
                "artifact_detection".to_string(),
                "sharpness_measurement".to_string(),
            ],
            quality_thresholds: {
                let mut thresholds = HashMap::new();
                thresholds.insert("noise_threshold".to_string(), 0.1);
                thresholds.insert("contrast_threshold".to_string(), 0.3);
                thresholds.insert("artifact_threshold".to_string(), 0.2);
                thresholds.insert("sharpness_threshold".to_string(), 0.5);
                thresholds
            },
        }
    }
    
    pub async fn assess_quality(&self, study_data: &crate::radiology_ai::StudyData) -> Result<crate::radiology_ai::QualityMetrics> {
        let mut noise_score = 0.0;
        let mut contrast_score = 0.0;
        let mut artifact_score = 0.0;
        let mut sharpness_score = 0.0;
        
        for image in &study_data.images {
            noise_score += self.assess_noise(image);
            contrast_score += self.assess_contrast(image);
            artifact_score += self.assess_artifacts(image);
            sharpness_score += self.assess_sharpness(image);
        }
        
        let image_count = study_data.images.len() as f64;
        noise_score /= image_count;
        contrast_score /= image_count;
        artifact_score /= image_count;
        sharpness_score /= image_count;
        
        let overall_score = (noise_score + contrast_score + artifact_score + sharpness_score) / 4.0;
        
        Ok(crate::radiology_ai::QualityMetrics {
            overall_score,
            noise_level: 1.0 - noise_score,
            contrast_score,
            artifact_score: 1.0 - artifact_score,
            sharpness_score,
        })
    }
    
    fn assess_noise(&self, image: &SparseMatrix<f64>) -> f64 {
        let values = image.values();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let noise_level = variance.sqrt() / mean.abs();
        1.0 - noise_level.min(1.0)
    }
    
    fn assess_contrast(&self, image: &SparseMatrix<f64>) -> f64 {
        let values = image.values();
        if values.is_empty() {
            return 0.0;
        }
        
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val > min_val {
            (max_val - min_val) / (max_val + min_val)
        } else {
            0.0
        }
    }
    
    fn assess_artifacts(&self, image: &SparseMatrix<f64>) -> f64 {
        // Simplified artifact detection based on value distribution
        let values = image.values();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let outliers = values.iter()
            .filter(|&&x| (x - mean).abs() > 3.0 * mean)
            .count();
        
        let artifact_ratio = outliers as f64 / values.len() as f64;
        1.0 - artifact_ratio.min(1.0)
    }
    
    fn assess_sharpness(&self, image: &SparseMatrix<f64>) -> f64 {
        // Simplified sharpness measurement using gradient magnitude
        let values = image.values();
        if values.len() < 2 {
            return 0.0;
        }
        
        let mut gradient_sum = 0.0;
        for i in 1..values.len() {
            gradient_sum += (values[i] - values[i-1]).abs();
        }
        
        let average_gradient = gradient_sum / (values.len() - 1) as f64;
        average_gradient.min(1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseOptimizationModule {
    pub optimization_algorithms: Vec<String>,
    pub dose_constraints: DoseConstraints,
}

impl DoseOptimizationModule {
    pub fn new() -> Self {
        Self {
            optimization_algorithms: vec![
                "ctdi_optimization".to_string(),
                "dlp_minimization".to_string(),
                "noise_index_optimization".to_string(),
            ],
            dose_constraints: DoseConstraints::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseConstraints {
    pub max_ctdi: f64,
    pub max_dlp: f64,
    pub target_noise_index: f64,
}

impl Default for DoseConstraints {
    fn default() -> Self {
        Self {
            max_ctdi: 75.0,  // mGy
            max_dlp: 1000.0, // mGy*cm
            target_noise_index: 10.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowIntegration {
    pub supported_systems: Vec<String>,
    pub integration_protocols: HashMap<String, String>,
}

impl WorkflowIntegration {
    pub fn new() -> Self {
        let mut protocols = HashMap::new();
        protocols.insert("PACS".to_string(), "DICOM".to_string());
        protocols.insert("RIS".to_string(), "HL7_FHIR".to_string());
        protocols.insert("EMR".to_string(), "HL7_FHIR".to_string());
        
        Self {
            supported_systems: vec![
                "PACS".to_string(),
                "RIS".to_string(),
                "EMR".to_string(),
                "Worklist".to_string(),
            ],
            integration_protocols: protocols,
        }
    }
}

// Preprocessing and enhancement types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingStep {
    BeamHardeningCorrection,
    ScatterCorrection,
    NoiseReduction,
    MotionCorrection,
    B0FieldCorrection,
    BiasFieldCorrection,
    FlatFieldCorrection,
    DeadPixelCorrection,
    GeometricCorrection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancementAlgorithm {
    ContrastEnhancement,
    EdgeEnhancement,
    NoiseReduction,
    IntensityNormalization,
    HistogramEqualization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactReductionModule {
    pub modality: String,
    pub reduction_algorithms: Vec<String>,
}

impl ArtifactReductionModule {
    pub fn new(modality: &str) -> Self {
        let algorithms = match modality {
            "CT" => vec![
                "metal_artifact_reduction".to_string(),
                "beam_hardening_correction".to_string(),
                "scatter_correction".to_string(),
            ],
            "MRI" => vec![
                "motion_artifact_reduction".to_string(),
                "susceptibility_artifact_correction".to_string(),
                "chemical_shift_correction".to_string(),
            ],
            "X-Ray" => vec![
                "grid_artifact_removal".to_string(),
                "detector_artifact_correction".to_string(),
                "geometry_correction".to_string(),
            ],
            _ => vec![],
        };
        
        Self {
            modality: modality.to_string(),
            reduction_algorithms: algorithms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOptimizer {
    pub modality: String,
    pub optimization_parameters: HashMap<String, f64>,
}

impl ProtocolOptimizer {
    pub fn new(modality: &str) -> Self {
        let mut parameters = HashMap::new();
        
        match modality {
            "CT" => {
                parameters.insert("kvp".to_string(), 120.0);
                parameters.insert("mas".to_string(), 200.0);
                parameters.insert("slice_thickness".to_string(), 2.5);
                parameters.insert("pitch".to_string(), 1.0);
            },
            "MRI" => {
                parameters.insert("tr".to_string(), 500.0);
                parameters.insert("te".to_string(), 20.0);
                parameters.insert("flip_angle".to_string(), 90.0);
                parameters.insert("slice_thickness".to_string(), 5.0);
            },
            "X-Ray" => {
                parameters.insert("kvp".to_string(), 100.0);
                parameters.insert("mas".to_string(), 10.0);
                parameters.insert("filtration".to_string(), 2.5);
            },
            _ => {},
        }
        
        Self {
            modality: modality.to_string(),
            optimization_parameters: parameters,
        }
    }
}

// Reconstruction algorithm types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IterativeMethod {
    None,
    OSEM,
    CGLS,
    MLEM,
    SART,
    ART,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    pub regularization_type: String,
    pub lambda: f64,
    pub iterations: u32,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            regularization_type: "Tikhonov".to_string(),
            lambda: 0.01,
            iterations: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    pub max_iterations: u32,
    pub tolerance: f64,
    pub relative_change_threshold: f64,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            relative_change_threshold: 1e-4,
        }
    }
}

// CAD System types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionAlgorithm {
    pub algorithm_name: String,
    pub sensitivity: f64,
    pub false_positive_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationModel {
    pub model_name: String,
    pub classes: Vec<String>,
    pub performance_metrics: crate::advanced_features::PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationEngine {
    pub engine_name: String,
    pub segmentation_type: String,
    pub dice_coefficient: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantitativeAnalysis {
    pub measurements: Vec<String>,
    pub radiomics_features: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingSystem {
    pub structured_reporting: bool,
    pub integration_with_ris: bool,
    pub automated_measurements: bool,
}

// Clinical validation types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalValidation {
    pub validation_studies: Vec<String>,
    pub patient_cohort_size: u32,
    pub validation_auc: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FDAClearance {
    pub clearance_number: String,
    pub clearance_date: String,
    pub intended_use: String,
}

// Advanced imaging types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamHardeningCorrection {
    pub correction_algorithm: String,
    pub calibration_data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterCorrection {
    pub scatter_model: String,
    pub correction_factors: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseReduction {
    pub noise_model: String,
    pub filter_parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalArtifactReduction {
    pub detection_threshold: f64,
    pub correction_algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelImagingConfig {
    pub acceleration_factor: f64,
    pub coil_geometry: String,
    pub reconstruction_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedSensingConfig {
    pub undersampling_pattern: String,
    pub sparsity_transform: String,
    pub regularization_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionCorrection {
    pub motion_detection_method: String,
    pub correction_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct B0FieldCorrection {
    pub field_map_acquisition: String,
    pub correction_method: String,
}

// Specific finding types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LungNodule {
    pub nodule_id: String,
    pub location: AnatomicalLocation,
    pub volume: f64,
    pub diameter: f64,
    pub density: f64,
    pub malignancy_score: f64,
    pub shape_features: ShapeFeatures,
    pub texture_features: TextureFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeFeatures {
    pub sphericity: f64,
    pub compactness: f64,
    pub elongation: f64,
    pub surface_area: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextureFeatures {
    pub entropy: f64,
    pub contrast: f64,
    pub homogeneity: f64,
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainLesionAnalysis {
    pub brain_segmentation: BrainSegmentation,
    pub detected_lesions: Vec<BrainLesion>,
    pub longitudinal_changes: LongitudinalAnalysis,
    pub radiomics_features: RadiomicsFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainSegmentation {
    pub gray_matter_mask: SparseMatrix<f64>,
    pub white_matter_mask: SparseMatrix<f64>,
    pub csf_mask: SparseMatrix<f64>,
    pub tissue_volumes: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainLesion {
    pub lesion_id: String,
    pub lesion_type: String,
    pub location: AnatomicalLocation,
    pub volume: f64,
    pub t1_characteristics: T1Characteristics,
    pub t2_characteristics: T2Characteristics,
    pub enhancement_pattern: EnhancementPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T1Characteristics {
    pub signal_intensity: String,
    pub contrast_enhancement: bool,
    pub enhancement_degree: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T2Characteristics {
    pub signal_intensity: String,
    pub hyperintense_areas: f64,
    pub hypointense_areas: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementPattern {
    pub pattern_type: String,
    pub rim_enhancement: bool,
    pub heterogeneous_enhancement: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongitudinalAnalysis {
    pub previous_studies: Vec<String>,
    pub volume_changes: HashMap<String, f64>,
    pub new_lesions: Vec<String>,
    pub resolved_lesions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadiomicsFeatures {
    pub first_order_features: HashMap<String, f64>,
    pub shape_features: HashMap<String, f64>,
    pub texture_features: HashMap<String, f64>,
    pub wavelet_features: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MammographyReport {
    pub breast_density: BreastDensity,
    pub detected_masses: Vec<BreastMass>,
    pub detected_calcifications: Vec<Calcification>,
    pub architectural_distortions: Vec<ArchitecturalDistortion>,
    pub birads_category: BIRADSCategory,
    pub recommendation: ClinicalRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreastDensity {
    pub density_category: String,
    pub density_percentage: f64,
    pub fibroglandular_tissue_volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreastMass {
    pub mass_id: String,
    pub location: AnatomicalLocation,
    pub size: f64,
    pub shape: String,
    pub margins: String,
    pub density: String,
    pub malignancy_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calcification {
    pub calcification_id: String,
    pub location: AnatomicalLocation,
    pub morphology: String,
    pub distribution: String,
    pub number: u32,
    pub malignancy_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalDistortion {
    pub distortion_id: String,
    pub location: AnatomicalLocation,
    pub severity: String,
    pub spiculation_pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BIRADSCategory {
    pub category: u8,
    pub description: String,
    pub follow_up_recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalRecommendation {
    pub recommendation_type: String,
    pub urgency: String,
    pub timeframe: String,
    pub details: Option<String>,
}

// Implementation helper for CADResults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CADResult {
    pub findings: Vec<ImagingFinding>,
    pub confidence_scores: HashMap<String, f64>,
    pub processing_time: f64,
}

impl CADResult {
    pub fn calculate_confidence(&self) -> f64 {
        if self.confidence_scores.is_empty() {
            return 0.0;
        }
        
        self.confidence_scores.values().sum::<f64>() / self.confidence_scores.len() as f64
    }
    
    pub fn format_findings(&self) -> String {
        self.findings.iter()
            .map(|finding| format!("{}: {}", finding.finding_type, finding.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// Implementation for radiology CADResults
impl crate::radiology_ai::CADResults {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }
    
    pub fn add_result(&mut self, cad_name: String, result: CADResult) {
        self.results.insert(cad_name, result);
    }
    
    pub fn findings(&self) -> Vec<ImagingFinding> {
        self.results.values()
            .flat_map(|result| result.findings.clone())
            .collect()
    }
} 