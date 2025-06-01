use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use chrono::{DateTime, Utc};

/// Medical imaging modalities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Modality {
    CT,
    MRI,
    PET,
    XRay,
    Ultrasound,
    Mammography,
    OCT, // Optical Coherence Tomography
    SPECT,
    Nuclear,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modality::CT => write!(f, "CT"),
            Modality::MRI => write!(f, "MRI"),
            Modality::PET => write!(f, "PET"),
            Modality::XRay => write!(f, "X-Ray"),
            Modality::Ultrasound => write!(f, "Ultrasound"),
            Modality::Mammography => write!(f, "Mammography"),
            Modality::OCT => write!(f, "OCT"),
            Modality::SPECT => write!(f, "SPECT"),
            Modality::Nuclear => write!(f, "Nuclear"),
        }
    }
}

/// Anatomical regions for medical AI
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnatomicalRegion {
    Brain,
    Heart,
    Lungs,
    Liver,
    Kidneys,
    Spine,
    Abdomen,
    Pelvis,
    Chest,
    Head,
    Neck,
    Extremities,
    WholeBody,
}

impl std::fmt::Display for AnatomicalRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnatomicalRegion::Brain => write!(f, "Brain"),
            AnatomicalRegion::Heart => write!(f, "Heart"),
            AnatomicalRegion::Lungs => write!(f, "Lungs"),
            AnatomicalRegion::Liver => write!(f, "Liver"),
            AnatomicalRegion::Kidneys => write!(f, "Kidneys"),
            AnatomicalRegion::Spine => write!(f, "Spine"),
            AnatomicalRegion::Abdomen => write!(f, "Abdomen"),
            AnatomicalRegion::Pelvis => write!(f, "Pelvis"),
            AnatomicalRegion::Chest => write!(f, "Chest"),
            AnatomicalRegion::Head => write!(f, "Head"),
            AnatomicalRegion::Neck => write!(f, "Neck"),
            AnatomicalRegion::Extremities => write!(f, "Extremities"),
            AnatomicalRegion::WholeBody => write!(f, "Whole Body"),
        }
    }
}

/// Medical AI tasks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MedicalTask {
    Segmentation,
    Classification,
    Detection,
    Regression,
    Registration,
    Reconstruction,
    Synthesis,
    Enhancement,
}

impl std::fmt::Display for MedicalTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MedicalTask::Segmentation => write!(f, "Segmentation"),
            MedicalTask::Classification => write!(f, "Classification"),
            MedicalTask::Detection => write!(f, "Detection"),
            MedicalTask::Regression => write!(f, "Regression"),
            MedicalTask::Registration => write!(f, "Registration"),
            MedicalTask::Reconstruction => write!(f, "Reconstruction"),
            MedicalTask::Synthesis => write!(f, "Synthesis"),
            MedicalTask::Enhancement => write!(f, "Enhancement"),
        }
    }
}

/// Clinical validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalMetrics {
    pub dice_coefficient: f64,
    pub jaccard_index: f64,
    pub hausdorff_distance: f64,
    pub sensitivity: f64,
    pub specificity: f64,
    pub positive_predictive_value: f64,
    pub negative_predictive_value: f64,
    pub accuracy: f64,
    pub auc_roc: f64,
    pub auc_pr: f64,
}

impl Default for ClinicalMetrics {
    fn default() -> Self {
        Self {
            dice_coefficient: 0.0,
            jaccard_index: 0.0,
            hausdorff_distance: 0.0,
            sensitivity: 0.0,
            specificity: 0.0,
            positive_predictive_value: 0.0,
            negative_predictive_value: 0.0,
            accuracy: 0.0,
            auc_roc: 0.0,
            auc_pr: 0.0,
        }
    }
}

impl ClinicalMetrics {
    /// Compute metrics from confusion matrix components
    pub fn from_confusion_matrix(tp: f64, tn: f64, fp: f64, fn_: f64) -> Self {
        let sensitivity = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let specificity = if tn + fp > 0.0 { tn / (tn + fp) } else { 0.0 };
        let ppv = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let npv = if tn + fn_ > 0.0 { tn / (tn + fn_) } else { 0.0 };
        let accuracy = (tp + tn) / (tp + tn + fp + fn_);
        let dice = if 2.0 * tp + fp + fn_ > 0.0 { 2.0 * tp / (2.0 * tp + fp + fn_) } else { 0.0 };
        let jaccard = if tp + fp + fn_ > 0.0 { tp / (tp + fp + fn_) } else { 0.0 };

        Self {
            dice_coefficient: dice,
            jaccard_index: jaccard,
            hausdorff_distance: 0.0, // Would need specialized computation
            sensitivity,
            specificity,
            positive_predictive_value: ppv,
            negative_predictive_value: npv,
            accuracy,
            auc_roc: 0.0, // Would need specialized computation
            auc_pr: 0.0,  // Would need specialized computation
        }
    }

    /// Check if metrics meet clinical standards
    pub fn meets_clinical_standards(&self, task: &MedicalTask) -> bool {
        match task {
            MedicalTask::Segmentation => {
                self.dice_coefficient >= 0.8 && 
                self.sensitivity >= 0.85 && 
                self.specificity >= 0.85
            },
            MedicalTask::Classification => {
                self.accuracy >= 0.9 && 
                self.sensitivity >= 0.85 && 
                self.specificity >= 0.85
            },
            MedicalTask::Detection => {
                self.sensitivity >= 0.9 && 
                self.positive_predictive_value >= 0.8
            },
            _ => self.accuracy >= 0.8, // General threshold
        }
    }
}

/// Medical image preprocessing utilities
pub struct MedicalImageProcessor;

impl MedicalImageProcessor {
    /// Normalize medical image intensities
    pub fn normalize_intensities(
        data: &mut [f64], 
        min_percentile: f64, 
        max_percentile: f64
    ) -> Result<()> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let min_idx = ((min_percentile / 100.0) * sorted_data.len() as f64) as usize;
        let max_idx = ((max_percentile / 100.0) * sorted_data.len() as f64) as usize;
        
        let min_val = sorted_data[min_idx.min(sorted_data.len() - 1)];
        let max_val = sorted_data[max_idx.min(sorted_data.len() - 1)];
        
        let range = max_val - min_val;
        if range > 0.0 {
            for value in data.iter_mut() {
                *value = (*value - min_val) / range;
                *value = value.max(0.0).min(1.0); // Clamp to [0, 1]
            }
        }
        
        Ok(())
    }

    /// Apply medical windowing (contrast adjustment)
    pub fn apply_windowing(
        data: &mut [f64], 
        window_center: f64, 
        window_width: f64
    ) -> Result<()> {
        let min_val = window_center - window_width / 2.0;
        let max_val = window_center + window_width / 2.0;
        
        for value in data.iter_mut() {
            if *value < min_val {
                *value = min_val;
            } else if *value > max_val {
                *value = max_val;
            }
        }
        
        Ok(())
    }

    /// Remove noise using median filtering
    pub fn median_filter_3d(
        data: &[f64], 
        shape: (usize, usize, usize), 
        kernel_size: usize
    ) -> Result<Vec<f64>> {
        let (nx, ny, nz) = shape;
        let mut filtered = vec![0.0; data.len()];
        let half_kernel = kernel_size / 2;
        
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mut neighborhood = Vec::new();
                    
                    for dz in 0..kernel_size {
                        for dy in 0..kernel_size {
                            for dx in 0..kernel_size {
                                let nz_idx = z + dz - half_kernel;
                                let ny_idx = y + dy - half_kernel;
                                let nx_idx = x + dx - half_kernel;
                                
                                if nz_idx < nz && ny_idx < ny && nx_idx < nx {
                                    let idx = nz_idx * nx * ny + ny_idx * nx + nx_idx;
                                    if idx < data.len() {
                                        neighborhood.push(data[idx]);
                                    }
                                }
                            }
                        }
                    }
                    
                    neighborhood.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = neighborhood[neighborhood.len() / 2];
                    
                    let idx = z * nx * ny + y * nx + x;
                    filtered[idx] = median;
                }
            }
        }
        
        Ok(filtered)
    }

    /// Extract texture features using Gray Level Co-occurrence Matrix
    pub fn extract_glcm_features(
        data: &[f64], 
        shape: (usize, usize), 
        num_levels: usize
    ) -> Result<HashMap<String, f64>> {
        let (height, width) = shape;
        let mut features = HashMap::new();
        
        // Quantize the image
        let max_val = data.iter().fold(0.0, |a, &b| a.max(b));
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_val - min_val;
        
        let quantized: Vec<usize> = data.iter()
            .map(|&x| ((x - min_val) / range * (num_levels - 1) as f64).round() as usize)
            .collect();
        
        // Build GLCM for horizontal direction (offset = 1, 0)
        let mut glcm = vec![vec![0; num_levels]; num_levels];
        
        for y in 0..height {
            for x in 0..(width - 1) {
                let i = quantized[y * width + x];
                let j = quantized[y * width + x + 1];
                glcm[i][j] += 1;
                glcm[j][i] += 1; // Make symmetric
            }
        }
        
        // Normalize GLCM
        let total: u32 = glcm.iter().flatten().sum();
        let glcm_normalized: Vec<Vec<f64>> = glcm.iter()
            .map(|row| row.iter().map(|&x| x as f64 / total as f64).collect())
            .collect();
        
        // Calculate texture features
        let mut contrast = 0.0;
        let mut homogeneity = 0.0;
        let mut entropy = 0.0;
        let mut energy = 0.0;
        
        for i in 0..num_levels {
            for j in 0..num_levels {
                let p = glcm_normalized[i][j];
                if p > 0.0 {
                    contrast += (i as f64 - j as f64).powi(2) * p;
                    homogeneity += p / (1.0 + (i as f64 - j as f64).abs());
                    entropy -= p * p.ln();
                    energy += p * p;
                }
            }
        }
        
        features.insert("contrast".to_string(), contrast);
        features.insert("homogeneity".to_string(), homogeneity);
        features.insert("entropy".to_string(), entropy);
        features.insert("energy".to_string(), energy);
        
        Ok(features)
    }
}

/// Medical data validation
pub struct MedicalValidator;

impl MedicalValidator {
    /// Validate DICOM metadata for clinical use
    pub fn validate_dicom_metadata(
        patient_id: &str,
        study_date: &str,
        modality: &str,
        anatomy: &str,
    ) -> Result<bool> {
        // Check required fields are not empty
        if patient_id.is_empty() || study_date.is_empty() || 
           modality.is_empty() || anatomy.is_empty() {
            return Ok(false);
        }
        
        // Validate study date format (YYYYMMDD)
        if study_date.len() != 8 || !study_date.chars().all(char::is_numeric) {
            return Ok(false);
        }
        
        // Validate modality is known
        let valid_modalities = ["CT", "MRI", "PET", "XR", "US", "MG", "OCT", "NM"];
        if !valid_modalities.contains(&modality) {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Validate image dimensions and voxel spacing
    pub fn validate_image_geometry(
        dimensions: &[usize],
        voxel_spacing: &[f64],
    ) -> Result<bool> {
        // Check dimensions match voxel spacing
        if dimensions.len() != voxel_spacing.len() {
            return Ok(false);
        }
        
        // Check for reasonable dimensions (not too small or too large)
        for &dim in dimensions {
            if dim < 16 || dim > 2048 {
                return Ok(false);
            }
        }
        
        // Check for reasonable voxel spacing (0.1mm to 10mm)
        for &spacing in voxel_spacing {
            if spacing < 0.1 || spacing > 10.0 {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Validate clinical metrics meet standards
    pub fn validate_clinical_performance(
        metrics: &ClinicalMetrics,
        task: &MedicalTask,
        anatomy: &AnatomicalRegion,
    ) -> Result<bool> {
        // Different standards for different anatomical regions
        let required_dice = match anatomy {
            AnatomicalRegion::Brain => 0.85,
            AnatomicalRegion::Heart => 0.80,
            AnatomicalRegion::Liver => 0.82,
            _ => 0.75,
        };
        
        let required_sensitivity = match task {
            MedicalTask::Detection => 0.90,
            MedicalTask::Classification => 0.85,
            _ => 0.80,
        };
        
        Ok(metrics.dice_coefficient >= required_dice && 
           metrics.sensitivity >= required_sensitivity &&
           metrics.specificity >= 0.80)
    }
}

/// Privacy and anonymization utilities
pub struct MedicalPrivacy;

impl MedicalPrivacy {
    /// Anonymize patient ID
    pub fn anonymize_patient_id(patient_id: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        patient_id.hash(&mut hasher);
        format!("ANON_{:x}", hasher.finish())
    }
    
    /// Remove identifying information from medical metadata
    pub fn anonymize_metadata(
        patient_id: &str,
        patient_name: Option<&str>,
        study_date: &str,
        preserve_temporal_info: bool,
    ) -> HashMap<String, String> {
        let mut anonymized = HashMap::new();
        
        // Always anonymize patient ID
        anonymized.insert(
            "patient_id".to_string(), 
            Self::anonymize_patient_id(patient_id)
        );
        
        // Remove patient name
        if patient_name.is_some() {
            anonymized.insert("patient_name".to_string(), "ANONYMOUS".to_string());
        }
        
        // Optionally preserve temporal information (shift dates)
        if preserve_temporal_info {
            // In practice, you'd implement date shifting while preserving intervals
            anonymized.insert("study_date".to_string(), "20240101".to_string());
        } else {
            anonymized.insert("study_date".to_string(), "REMOVED".to_string());
        }
        
        anonymized
    }
    
    /// Check if data has been properly anonymized
    pub fn verify_anonymization(metadata: &HashMap<String, String>) -> Result<bool> {
        // Check for common identifying patterns
        let identifying_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b", // SSN
            r"\b[A-Za-z]+,\s*[A-Za-z]+\b", // Name patterns
            r"\b\d{1,2}/\d{1,2}/\d{4}\b", // Date patterns
        ];
        
        for (_, value) in metadata {
            for pattern in &identifying_patterns {
                let regex = regex::Regex::new(pattern)?;
                if regex.is_match(value) {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clinical_metrics() {
        let metrics = ClinicalMetrics::from_confusion_matrix(85.0, 90.0, 10.0, 15.0);
        assert!(metrics.dice_coefficient > 0.8);
        assert!(metrics.sensitivity > 0.8);
        assert!(metrics.specificity > 0.8);
    }

    #[test]
    fn test_medical_validator() {
        assert!(MedicalValidator::validate_dicom_metadata(
            "PAT001", "20240101", "MRI", "Brain"
        ).unwrap());
        
        assert!(!MedicalValidator::validate_dicom_metadata(
            "", "20240101", "MRI", "Brain"
        ).unwrap());
    }

    #[test]
    fn test_anonymization() {
        let anon_id = MedicalPrivacy::anonymize_patient_id("PATIENT001");
        assert!(anon_id.starts_with("ANON_"));
        assert_ne!(anon_id, "PATIENT001");
    }
} 