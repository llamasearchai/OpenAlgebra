//! Neurology AI Engine for neuroimaging analysis and neurological disorder detection

use crate::{MedicalAgent, SparseMatrix, ApiError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainScan {
    pub patient_id: String,
    pub scan_data: SparseMatrix,
    pub modality: String, // "MRI", "CT", "fMRI"
    pub voxel_spacing: [f64; 3],
    pub contrast_agent: bool,
    pub clinical_history: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainAnalysis {
    pub lesions: Vec<Lesion>,
    pub volumetric_analysis: HashMap<String, f64>,
    pub functional_connectivity: SparseMatrix,
    pub stroke_risk: f64,
    pub neurodegeneration_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesion {
    pub location: [usize; 3],
    pub volume_cm3: f64,
    pub intensity_features: Vec<f64>,
    pub suspected_etiology: String,
}

pub struct NeurologyAIEngine {
    structural_model: SparseMatrix,
    functional_model: SparseMatrix,
    stroke_model: SparseMatrix,
    medical_agent: Arc<MedicalAgent>,
}

impl NeurologyAIEngine {
    pub fn new(medical_agent: MedicalAgent) -> Result<Self, ApiError> {
        Ok(Self {
            structural_model: SparseMatrix::load("models/neuro_structural.bin")?,
            functional_model: SparseMatrix::load("models/neuro_functional.bin")?,
            stroke_model: SparseMatrix::load("models/neuro_stroke.bin")?,
            medical_agent: Arc::new(medical_agent),
        })
    }

    pub async fn analyze_brain(&self, scan: &BrainScan) -> Result<BrainAnalysis, ApiError> {
        let structural_features = self.extract_structural_features(scan)?;
        let functional_features = self.extract_functional_features(scan)?;
        
        Ok(BrainAnalysis {
            lesions: self.detect_lesions(&structural_features)?,
            volumetric_analysis: self.calculate_volumetrics(&structural_features)?,
            functional_connectivity: self.analyze_connectivity(&functional_features)?,
            stroke_risk: self.predict_stroke_risk(&structural_features, &functional_features)?,
            neurodegeneration_score: self.calculate_neurodegeneration_score(&structural_features)?,
        })
    }

    fn extract_structural_features(&self, scan: &BrainScan) -> Result<Vec<f64>, ApiError> {
        let mut features = Vec::new();
        let density = scan.scan_data.density();
        features.push(density);
        features.push(scan.voxel_spacing[0]);
        features.push(scan.voxel_spacing[1]);
        features.push(scan.voxel_spacing[2]);
        if scan.contrast_agent {
            features.push(1.0);
        } else {
            features.push(0.0);
        }
        Ok(features)
    }

    fn extract_functional_features(&self, scan: &BrainScan) -> Result<Vec<f64>, ApiError> {
        let mut features = Vec::new();
        if scan.modality == "fMRI" {
            features.push(1.0);
            features.extend_from_slice(&self.extract_structural_features(scan)?);
        } else {
            features.push(0.0);
            features.extend(vec![0.0; 5]);
        }
        Ok(features)
    }

    fn detect_lesions(&self, features: &[f64]) -> Result<Vec<Lesion>, ApiError> {
        let predictions = self.structural_model.predict(features)?;
        let mut lesions = Vec::new();
        if predictions[0] > 0.5 {
            lesions.push(Lesion {
                location: [100, 100, 50],
                volume_cm3: predictions[1] * 10.0,
                intensity_features: predictions[2..5].to_vec(),
                suspected_etiology: "Ischemic".into(),
            });
        }
        Ok(lesions)
    }

    fn calculate_volumetrics(&self, features: &[f64]) -> Result<HashMap<String, f64>, ApiError> {
        let predictions = self.structural_model.predict(features)?;
        let mut volumetrics = HashMap::new();
        volumetrics.insert("Hippocampus".into(), predictions[5] * 5.0);
        volumetrics.insert("Cortex".into(), predictions[6] * 100.0);
        volumetrics.insert("Ventricles".into(), predictions[7] * 20.0);
        Ok(volumetrics)
    }

    fn analyze_connectivity(&self, features: &[f64]) -> Result<SparseMatrix, ApiError> {
        self.functional_model.predict_matrix(features)
    }

    fn predict_stroke_risk(&self, structural_features: &[f64], functional_features: &[f64]) -> Result<f64, ApiError> {
        let mut combined = Vec::new();
        combined.extend_from_slice(structural_features);
        combined.extend_from_slice(functional_features);
        let risk = self.stroke_model.predict_single(&combined)?;
        Ok(risk)
    }

    fn calculate_neurodegeneration_score(&self, features: &[f64]) -> Result<f64, ApiError> {
        let score = self.structural_model.predict_single(features)?;
        Ok(score * 100.0)
    }
} 