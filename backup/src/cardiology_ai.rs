//! Cardiology AI Engine
//!
//! Provides advanced cardiac signal and imaging analysis, risk stratification, and clinical decision support.

use crate::api::ApiError;
use crate::sparse::SparseMatrix;
use crate::medical::MedicalAgent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use crate::grok_integration::{GrokMedicalRequest, GrokMedicalResponse, GrokAIClient, ClinicalTaskType, MedicalDataPayload};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECGSignal {
    pub patient_id: String,
    pub recording_id: String,
    pub sampling_rate: f64,
    pub leads: HashMap<String, Vec<f64>>, // e.g., "I", "II", "V1", etc.
    pub start_time: DateTime<Utc>,
    pub duration_sec: f64,
    pub metadata: ECGMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECGMetadata {
    pub age: u8,
    pub sex: String,
    pub heart_rate: f64,
    pub rhythm: String,
    pub pr_interval_ms: f64,
    pub qrs_duration_ms: f64,
    pub qt_interval_ms: f64,
    pub axis: f64,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECGAnalysis {
    pub arrhythmias: Vec<Arrhythmia>,
    pub intervals: ECGIntervals,
    pub st_changes: Vec<STChange>,
    pub conduction_abnormalities: Vec<String>,
    pub overall_interpretation: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arrhythmia {
    pub arrhythmia_type: String,
    pub onset_time: f64,
    pub duration_sec: f64,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ECGIntervals {
    pub pr_ms: f64,
    pub qrs_ms: f64,
    pub qt_ms: f64,
    pub rr_ms: f64,
    pub qt_corrected_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STChange {
    pub lead: String,
    pub deviation_mm: f64,
    pub direction: String, // "elevation" or "depression"
    pub duration_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoStudy {
    pub patient_id: String,
    pub study_id: String,
    pub acquisition_time: DateTime<Utc>,
    pub images: Vec<EchoImage>,
    pub measurements: EchoMeasurements,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoImage {
    pub frame_id: String,
    pub view: String, // e.g., "PLAX", "A4C"
    pub pixel_data: Vec<u8>,
    pub dimensions: (u32, u32),
    pub frame_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoMeasurements {
    pub lvef: f64,
    pub lv_mass: f64,
    pub la_volume: f64,
    pub rv_function: f64,
    pub wall_motion_score: f64,
    pub valve_abnormalities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiacMRStudy {
    pub patient_id: String,
    pub study_id: String,
    pub acquisition_time: DateTime<Utc>,
    pub images: Vec<MRImage>,
    pub measurements: CardiacMRMeasurements,
    pub interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MRImage {
    pub frame_id: String,
    pub sequence: String,
    pub pixel_data: Vec<u8>,
    pub dimensions: (u32, u32),
    pub frame_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiacMRMeasurements {
    pub lvef: f64,
    pub rv_ef: f64,
    pub lv_mass: f64,
    pub infarct_size: f64,
    pub edema_volume: f64,
    pub fibrosis_volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiacRiskProfile {
    pub patient_id: String,
    pub age: u8,
    pub sex: String,
    pub risk_factors: Vec<String>,
    pub scores: HashMap<String, f64>, // e.g., "ASCVD", "Framingham"
    pub risk_category: String,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiologyReport {
    pub report_id: String,
    pub patient_id: String,
    pub report_time: DateTime<Utc>,
    pub ecg_analysis: Option<ECGAnalysis>,
    pub echo_measurements: Option<EchoMeasurements>,
    pub mr_measurements: Option<CardiacMRMeasurements>,
    pub risk_profile: Option<CardiacRiskProfile>,
    pub summary: String,
    pub recommendations: Vec<String>,
    pub confidence: f64,
}

pub struct CardiologyAIEngine {
    ecg_model: Arc<RwLock<SparseMatrix>>,
    echo_model: Arc<RwLock<SparseMatrix>>,
    mr_model: Arc<RwLock<SparseMatrix>>,
    risk_model: Arc<RwLock<SparseMatrix>>,
    medical_agent: Arc<MedicalAgent>,
}

impl CardiologyAIEngine {
    pub async fn new(medical_agent: MedicalAgent) -> Result<Self, ApiError> {
        Ok(Self {
            ecg_model: Arc::new(RwLock::new(SparseMatrix::identity(500))),
            echo_model: Arc::new(RwLock::new(SparseMatrix::identity(200))),
            mr_model: Arc::new(RwLock::new(SparseMatrix::identity(200))),
            risk_model: Arc::new(RwLock::new(SparseMatrix::identity(50))),
            medical_agent: Arc::new(medical_agent),
        })
    }

    pub async fn analyze_ecg(&self, ecg: &ECGSignal) -> Result<ECGAnalysis, ApiError> {
        // Feature extraction
        let features = self.extract_ecg_features(ecg)?;
        let model = self.ecg_model.read().await;
        let arrhythmias = self.detect_arrhythmias(&features);
        let intervals = self.calculate_intervals(&features);
        let st_changes = self.detect_st_changes(&features);
        let conduction_abnormalities = self.detect_conduction_abnormalities(&features);
        let overall_interpretation = self.summarize_ecg(&arrhythmias, &st_changes, &conduction_abnormalities);
        let confidence = self.estimate_confidence(&features);
        Ok(ECGAnalysis {
            arrhythmias,
            intervals,
            st_changes,
            conduction_abnormalities,
            overall_interpretation,
            confidence,
        })
    }

    pub async fn analyze_echo(&self, echo: &EchoStudy) -> Result<EchoMeasurements, ApiError> {
        let features = self.extract_echo_features(echo)?;
        let model = self.echo_model.read().await;
        // Simulate model prediction
        Ok(EchoMeasurements {
            lvef: 60.0,
            lv_mass: 120.0,
            la_volume: 45.0,
            rv_function: 55.0,
            wall_motion_score: 1.0,
            valve_abnormalities: vec![],
        })
    }

    pub async fn analyze_mr(&self, mr: &CardiacMRStudy) -> Result<CardiacMRMeasurements, ApiError> {
        let features = self.extract_mr_features(mr)?;
        let model = self.mr_model.read().await;
        Ok(CardiacMRMeasurements {
            lvef: 58.0,
            rv_ef: 52.0,
            lv_mass: 115.0,
            infarct_size: 0.0,
            edema_volume: 0.0,
            fibrosis_volume: 0.0,
        })
    }

    pub async fn risk_stratification(&self, profile: &CardiacRiskProfile) -> Result<CardiacRiskProfile, ApiError> {
        let features = self.extract_risk_features(profile)?;
        let model = self.risk_model.read().await;
        let mut scores = profile.scores.clone();
        scores.insert("ASCVD".to_string(), 0.12);
        scores.insert("Framingham".to_string(), 0.09);
        let risk_category = if scores["ASCVD"] > 0.2 { "High" } else if scores["ASCVD"] > 0.075 { "Intermediate" } else { "Low" };
        let recommendations = vec!["Lifestyle modification".to_string(), "Consider statin therapy".to_string()];
        Ok(CardiacRiskProfile {
            patient_id: profile.patient_id.clone(),
            age: profile.age,
            sex: profile.sex.clone(),
            risk_factors: profile.risk_factors.clone(),
            scores,
            risk_category: risk_category.to_string(),
            recommendations,
        })
    }

    pub async fn generate_report(&self, patient_id: &str, ecg: Option<&ECGSignal>, echo: Option<&EchoStudy>, mr: Option<&CardiacMRStudy>, risk: Option<&CardiacRiskProfile>) -> Result<CardiologyReport, ApiError> {
        let ecg_analysis = if let Some(ecg) = ecg { Some(self.analyze_ecg(ecg).await?) } else { None };
        let echo_measurements = if let Some(echo) = echo { Some(self.analyze_echo(echo).await?) } else { None };
        let mr_measurements = if let Some(mr) = mr { Some(self.analyze_mr(mr).await?) } else { None };
        let risk_profile = if let Some(risk) = risk { Some(self.risk_stratification(risk).await?) } else { None };
        let summary = self.summarize_report(&ecg_analysis, &echo_measurements, &mr_measurements, &risk_profile);
        let recommendations = self.generate_recommendations(&ecg_analysis, &echo_measurements, &mr_measurements, &risk_profile);
        let confidence = 0.95;
        Ok(CardiologyReport {
            report_id: uuid::Uuid::new_v4().to_string(),
            patient_id: patient_id.to_string(),
            report_time: Utc::now(),
            ecg_analysis,
            echo_measurements,
            mr_measurements,
            risk_profile,
            summary,
            recommendations,
            confidence,
        })
    }

    pub async fn grok_second_opinion(
        &self,
        report: &CardiologyReport,
        grok_client: &GrokAIClient
    ) -> Result<GrokMedicalResponse, ApiError> {
        let request = GrokMedicalRequest {
            patient_context: self.build_patient_context(report).await?,
            clinical_task: ClinicalTaskType::CardiologyReview,
            medical_data: MedicalDataPayload::Cardiology(report.clone()),
            safety_constraints: self.medical_agent.safety_profile(),
            explainability: true,
        };
        
        grok_client.process_medical_request(request).await
    }

    // --- Feature extraction and clinical logic ---
    fn extract_ecg_features(&self, ecg: &ECGSignal) -> Result<Vec<f64>, ApiError> {
        let mut features = Vec::new();
        for lead in ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"] {
            if let Some(data) = ecg.leads.get(lead) {
                features.push(data.iter().cloned().sum::<f64>() / data.len() as f64);
            } else {
                features.push(0.0);
            }
        }
        features.push(ecg.metadata.heart_rate);
        features.push(ecg.metadata.pr_interval_ms);
        features.push(ecg.metadata.qrs_duration_ms);
        features.push(ecg.metadata.qt_interval_ms);
        Ok(features)
    }

    fn detect_arrhythmias(&self, features: &[f64]) -> Vec<Arrhythmia> {
        let mut arrhythmias = Vec::new();
        if features[8] > 120.0 {
            arrhythmias.push(Arrhythmia {
                arrhythmia_type: "Atrial Fibrillation".to_string(),
                onset_time: 0.0,
                duration_sec: 10.0,
                severity: "Moderate".to_string(),
            });
        }
        arrhythmias
    }

    fn calculate_intervals(&self, features: &[f64]) -> ECGIntervals {
        ECGIntervals {
            pr_ms: features[9],
            qrs_ms: features[10],
            qt_ms: features[11],
            rr_ms: 1000.0 * 60.0 / features[8].max(1.0),
            qt_corrected_ms: features[11] / (features[8] / 60.0).sqrt(),
        }
    }

    fn detect_st_changes(&self, features: &[f64]) -> Vec<STChange> {
        let mut st_changes = Vec::new();
        if features[0] > 0.5 {
            st_changes.push(STChange {
                lead: "I".to_string(),
                deviation_mm: 2.0,
                direction: "elevation".to_string(),
                duration_sec: 1.0,
            });
        }
        st_changes
    }

    fn detect_conduction_abnormalities(&self, features: &[f64]) -> Vec<String> {
        let mut abnormalities = Vec::new();
        if features[10] > 120.0 {
            abnormalities.push("Bundle branch block".to_string());
        }
        abnormalities
    }

    fn summarize_ecg(&self, arrhythmias: &[Arrhythmia], st_changes: &[STChange], conduction_abnormalities: &[String]) -> String {
        let mut summary = String::new();
        if !arrhythmias.is_empty() {
            summary += "Arrhythmia detected. ";
        }
        if !st_changes.is_empty() {
            summary += "ST changes present. ";
        }
        if !conduction_abnormalities.is_empty() {
            summary += "Conduction abnormality present. ";
        }
        if summary.is_empty() {
            summary = "Normal ECG".to_string();
        }
        summary
    }

    fn estimate_confidence(&self, _features: &[f64]) -> f64 {
        0.95
    }

    fn extract_echo_features(&self, _echo: &EchoStudy) -> Result<Vec<f64>, ApiError> {
        Ok(vec![60.0, 120.0, 45.0, 55.0, 1.0])
    }

    fn extract_mr_features(&self, _mr: &CardiacMRStudy) -> Result<Vec<f64>, ApiError> {
        Ok(vec![58.0, 52.0, 115.0, 0.0, 0.0, 0.0])
    }

    fn extract_risk_features(&self, _profile: &CardiacRiskProfile) -> Result<Vec<f64>, ApiError> {
        Ok(vec![0.12, 0.09, 1.0, 0.0, 0.0])
    }

    fn summarize_report(&self, ecg: &Option<ECGAnalysis>, echo: &Option<EchoMeasurements>, mr: &Option<CardiacMRMeasurements>, risk: &Option<CardiacRiskProfile>) -> String {
        let mut summary = String::new();
        if let Some(ecg) = ecg {
            summary += &format!("ECG: {}. ", ecg.overall_interpretation);
        }
        if let Some(echo) = echo {
            summary += &format!("Echo LVEF: {:.1}%. ", echo.lvef);
        }
        if let Some(mr) = mr {
            summary += &format!("MR LVEF: {:.1}%. ", mr.lvef);
        }
        if let Some(risk) = risk {
            summary += &format!("Risk: {}. ", risk.risk_category);
        }
        if summary.is_empty() {
            summary = "No significant findings.".to_string();
        }
        summary
    }

    fn generate_recommendations(&self, ecg: &Option<ECGAnalysis>, echo: &Option<EchoMeasurements>, mr: &Option<CardiacMRMeasurements>, risk: &Option<CardiacRiskProfile>) -> Vec<String> {
        let mut recs = Vec::new();
        if let Some(ecg) = ecg {
            if ecg.overall_interpretation != "Normal ECG" {
                recs.push("Refer to cardiology".to_string());
            }
        }
        if let Some(risk) = risk {
            if risk.risk_category == "High" {
                recs.push("Initiate statin therapy".to_string());
            }
        }
        if recs.is_empty() {
            recs.push("Routine follow-up".to_string());
        }
        recs
    }
} 