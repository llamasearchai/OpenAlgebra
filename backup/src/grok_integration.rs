//! Grok Medical AI API integration with full clinical safety controls

use serde::{Serialize, Deserialize};
use reqwest::{Client, header};
use crate::{MedicalAgent, ApiError, SparseMatrix};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientContext {
    pub id: String,
    pub age: u8,
    pub sex: String,
    pub medical_history: Vec<String>,
    pub current_medications: Vec<String>,
    pub allergies: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClinicalTaskType {
    Diagnosis,
    TreatmentPlanning,
    Prognosis,
    SecondOpinion,
    CardiologyReview,
    OncologyAssessment,
    NeurologyEvaluation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MedicalDataPayload {
    RawData(Vec<f64>),
    ImagingData { dimensions: Vec<usize>, voxel_data: Vec<f64> },
    Cardiology(crate::cardiology_ai::CardiologyReport),
    Oncology(crate::oncology_ai::TumorAnalysis),
    Neurology(crate::neurology_ai::BrainAnalysis),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraints {
    pub max_risk_level: f64,
    pub required_confidence: f64,
    pub compliance_standards: Vec<String>,
    pub data_privacy_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalDecision {
    pub primary_diagnosis: String,
    pub treatment_recommendations: Vec<String>,
    pub follow_up_actions: Vec<String>,
    pub urgency_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialDiagnosis {
    pub condition: String,
    pub probability: f64,
    pub supporting_evidence: Vec<String>,
    pub ruling_out_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalEvidence {
    pub source: String,
    pub reference_id: String,
    pub relevance_score: f64,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyStatus {
    pub risk_level: f64,
    pub reasons: Vec<String>,
    pub mitigation_suggestions: Vec<String>,
    pub compliance_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrokMedicalRequest {
    pub patient_context: PatientContext,
    pub clinical_task: ClinicalTaskType,
    pub medical_data: MedicalDataPayload,
    pub safety_constraints: SafetyConstraints,
    pub explainability: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrokMedicalResponse {
    pub clinical_decision: ClinicalDecision,
    pub confidence_scores: HashMap<String, f64>,
    pub differentials: Vec<DifferentialDiagnosis>,
    pub evidence_references: Vec<MedicalEvidence>,
    pub safety_status: SafetyStatus,
    pub explanation: String,
}

#[derive(Debug, Clone)]
pub struct GrokAIClient {
    client: Client,
    base_url: String,
    api_key: String,
    medical_agent: Arc<MedicalAgent>,
    safety_checker: Arc<RwLock<SparseMatrix>>,
}

impl GrokAIClient {
    pub fn new(
        api_key: String,
        medical_agent: MedicalAgent,
        safety_model: SparseMatrix
    ) -> Result<Self, ApiError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            base_url: "https://api.grok.ai/medical/v1".into(),
            api_key,
            medical_agent: Arc::new(medical_agent),
            safety_checker: Arc::new(RwLock::new(safety_model)),
        })
    }

    pub async fn process_medical_request(
        &self,
        request: GrokMedicalRequest
    ) -> Result<GrokMedicalResponse, ApiError> {
        // Validate input with medical agent
        self.medical_agent.validate_request(&request)?;
        
        // Perform safety check
        let safety_features = self.extract_safety_features(&request)?;
        let safety_status = self.safety_checker.read().await.predict(&safety_features)?;
        
        if safety_status.risk_level > 0.7 {
            return Err(ApiError::SafetyViolation(safety_status.reasons));
        }

        // Build API request
        let response = self.client.post(&self.format_url("process"))
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header("X-Medical-Check", "strict")
            .json(&request)
            .send()
            .await?;

        // Process and validate response
        let mut response: GrokMedicalResponse = response.json().await?;
        self.validate_response(&mut response)?;
        
        Ok(response)
    }

    pub async fn get_diagnostic_insights(
        &self,
        patient_context: PatientContext,
        medical_data: MedicalDataPayload
    ) -> Result<GrokMedicalResponse, ApiError> {
        let request = GrokMedicalRequest {
            patient_context,
            clinical_task: ClinicalTaskType::Diagnosis,
            medical_data,
            safety_constraints: SafetyConstraints {
                max_risk_level: 0.5,
                required_confidence: 0.85,
                compliance_standards: vec!["HIPAA".into(), "GDPR".into()],
                data_privacy_level: "strict".into(),
            },
            explainability: true,
        };
        self.process_medical_request(request).await
    }

    pub async fn get_treatment_plan(
        &self,
        patient_context: PatientContext,
        medical_data: MedicalDataPayload
    ) -> Result<GrokMedicalResponse, ApiError> {
        let request = GrokMedicalRequest {
            patient_context,
            clinical_task: ClinicalTaskType::TreatmentPlanning,
            medical_data,
            safety_constraints: SafetyConstraints {
                max_risk_level: 0.4,
                required_confidence: 0.9,
                compliance_standards: vec!["HIPAA".into(), "GDPR".into()],
                data_privacy_level: "strict".into(),
            },
            explainability: true,
        };
        self.process_medical_request(request).await
    }

    pub async fn get_prognosis(
        &self,
        patient_context: PatientContext,
        medical_data: MedicalDataPayload
    ) -> Result<GrokMedicalResponse, ApiError> {
        let request = GrokMedicalRequest {
            patient_context,
            clinical_task: ClinicalTaskType::Prognosis,
            medical_data,
            safety_constraints: SafetyConstraints {
                max_risk_level: 0.6,
                required_confidence: 0.8,
                compliance_standards: vec!["HIPAA".into(), "GDPR".into()],
                data_privacy_level: "strict".into(),
            },
            explainability: true,
        };
        self.process_medical_request(request).await
    }

    fn format_url(&self, endpoint: &str) -> String {
        format!("{}/{}", self.base_url, endpoint)
    }

    fn extract_safety_features(&self, request: &GrokMedicalRequest) -> Result<Vec<f64>, ApiError> {
        let mut features = Vec::new();
        features.push(request.patient_context.age as f64);
        features.push(if request.patient_context.sex == "male" { 1.0 } else { 0.0 });
        features.push(request.patient_context.medical_history.len() as f64);
        features.push(request.patient_context.current_medications.len() as f64);
        features.push(request.patient_context.allergies.len() as f64);
        Ok(features)
    }

    fn validate_response(&self, response: &mut GrokMedicalResponse) -> Result<(), ApiError> {
        if response.safety_status.risk_level > 0.7 {
            return Err(ApiError::SafetyViolation(response.safety_status.reasons.clone()));
        }
        if response.confidence_scores.values().any(|&score| score < 0.5) {
            response.explanation += "\nWarning: Low confidence in some predictions.";
        }
        Ok(())
    }
} 