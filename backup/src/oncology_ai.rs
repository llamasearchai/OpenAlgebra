//! Oncology AI Engine for tumor analysis, treatment planning, and outcome prediction

use crate::{MedicalAgent, SparseMatrix, ApiError};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TumorScan {
    pub patient_id: String,
    pub scan_id: String,
    pub modality: String, // "CT", "MRI", "PET"
    pub voxel_data: Vec<f64>,
    pub dimensions: [usize; 3],
    pub voxel_spacing: [f64; 3],
    pub acquisition_date: DateTime<Utc>,
    pub biomarkers: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TumorAnalysis {
    pub volume_cm3: f64,
    pub stage: u8,
    pub tnm_classification: String,
    pub growth_rate: f64,
    pub metabolic_activity: f64,
    pub genetic_markers: HashSet<String>,
    pub treatment_resistance: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadiationPlan {
    pub target_volume: Vec<[usize; 3]>,
    pub dose_gy: f64,
    pub fractions: u16,
    pub organs_at_risk: HashMap<String, f64>,
    pub dose_distribution: SparseMatrix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemoRegimen {
    pub drugs: Vec<String>,
    pub doses: HashMap<String, f64>,
    pub cycle_days: u8,
    pub predicted_efficacy: f64,
    pub toxicity_risk: f64,
}

pub struct OncologyAIEngine {
    tumor_model: SparseMatrix,
    treatment_model: SparseMatrix,
    survival_model: SparseMatrix,
    medical_agent: Arc<MedicalAgent>,
}

impl OncologyAIEngine {
    pub fn new(medical_agent: MedicalAgent) -> Result<Self, ApiError> {
        // Model loading logic
        Ok(Self {
            tumor_model: SparseMatrix::load("models/oncology_tumor.bin")?,
            treatment_model: SparseMatrix::load("models/oncology_treatment.bin")?,
            survival_model: SparseMatrix::load("models/oncology_survival.bin")?,
            medical_agent: Arc::new(medical_agent),
        })
    }

    pub async fn analyze_tumor(&self, scan: &TumorScan) -> Result<TumorAnalysis, ApiError> {
        let features = self.extract_imaging_features(scan)?;
        let mut analysis = self.tumor_model.predict(&features)?;
        analysis.biomarkers = scan.biomarkers.clone();
        Ok(analysis)
    }

    pub async fn generate_radiation_plan(&self, analysis: &TumorAnalysis) -> Result<RadiationPlan, ApiError> {
        let input = self.prepare_radiation_input(analysis);
        self.treatment_model.solve_optimization(input)
    }

    pub async fn recommend_chemo(&self, analysis: &TumorAnalysis) -> Result<Vec<ChemoRegimen>, ApiError> {
        let mut regimens = self.treatment_model.predict_topn(&analysis.to_features(), 5)?;
        self.apply_safety_filters(&mut regimens)?;
        Ok(regimens)
    }

    pub async fn predict_survival(&self, analysis: &TumorAnalysis, treatment: &ChemoRegimen) -> Result<f64, ApiError> {
        let features = self.combine_features(analysis, treatment);
        self.survival_model.predict_single(&features)
    }

    fn extract_imaging_features(&self, scan: &TumorScan) -> Result<Vec<f64>, ApiError> {
        let mut features = Vec::new();
        let total_voxels = scan.dimensions[0] * scan.dimensions[1] * scan.dimensions[2];
        if scan.voxel_data.len() != total_voxels {
            return Err(ApiError::InvalidInput("Voxel data length mismatch".into()));
        }
        let mean_intensity = scan.voxel_data.iter().sum::<f64>() / total_voxels as f64;
        let variance = scan.voxel_data.iter().map(|v| (v - mean_intensity).powi(2)).sum::<f64>() / total_voxels as f64;
        features.push(mean_intensity);
        features.push(variance);
        features.push(scan.voxel_spacing[0]);
        features.push(scan.voxel_spacing[1]);
        features.push(scan.voxel_spacing[2]);
        for (_, value) in &scan.biomarkers {
            features.push(*value);
        }
        Ok(features)
    }

    fn prepare_radiation_input(&self, analysis: &TumorAnalysis) -> Vec<f64> {
        let mut input = Vec::new();
        input.push(analysis.volume_cm3);
        input.push(analysis.growth_rate);
        input.push(analysis.metabolic_activity);
        for resistance in analysis.treatment_resistance.values() {
            input.push(*resistance);
        }
        input
    }

    fn apply_safety_filters(&self, regimens: &mut Vec<ChemoRegimen>) -> Result<(), ApiError> {
        regimens.retain(|r| r.toxicity_risk < 0.5);
        if regimens.is_empty() {
            return Err(ApiError::NoSafeOptions("No safe chemotherapy regimens found".into()));
        }
        regimens.sort_by(|a, b| b.predicted_efficacy.partial_cmp(&a.predicted_efficacy).unwrap());
        Ok(())
    }

    fn combine_features(&self, analysis: &TumorAnalysis, treatment: &ChemoRegimen) -> Vec<f64> {
        let mut features = Vec::new();
        features.push(analysis.volume_cm3);
        features.push(analysis.stage as f64);
        features.push(analysis.growth_rate);
        features.push(analysis.metabolic_activity);
        features.push(treatment.predicted_efficacy);
        features.push(treatment.toxicity_risk);
        features.push(treatment.cycle_days as f64);
        for dose in treatment.doses.values() {
            features.push(*dose);
        }
        features
    }
} 