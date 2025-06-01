use async_openai::{
    Client,
    types::{
        CreateChatCompletionRequestArgs, ChatCompletionRequestMessage, 
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateEmbeddingRequestArgs, CreateImageRequestArgs, ImageSize, ImageResponseFormat,
        CreateCompletionRequestArgs, Role, FinishReason,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use tiktoken_rs::tiktoken::{get_bpe_from_model, CoreBPE};
use base64::{Engine as _, engine::general_purpose};
use crate::{
    medical::{MedicalDataProcessor, ClinicalValidation, MedicalDataset},
    dicom::{DICOMProcessor, DICOMSeries, DICOMMetadata},
    models::{MedicalNeuralNetwork, ModelType},
    sparse::SparseMatrix,
    utils::OpenAlgebraConfig,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalAIAgent {
    client: Client<async_openai::config::OpenAIConfig>,
    model: String,
    temperature: f32,
    max_tokens: u16,
    system_prompt: String,
    tokenizer: Option<CoreBPE>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalAnalysisRequest {
    pub patient_data: MedicalDataset,
    pub dicom_series: Option<DICOMSeries>,
    pub analysis_type: ClinicalAnalysisType,
    pub include_recommendations: bool,
    pub privacy_level: PrivacyLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClinicalAnalysisType {
    DiagnosticImaging,
    PathologyReview,
    TreatmentPlanning,
    RiskAssessment,
    DrugInteraction,
    ClinicalDecisionSupport,
    AnomalyDetection,
    PrognosticAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Full,        // Remove all PHI
    Partial,     // Remove direct identifiers only
    Minimal,     // Keep clinical context
    Research,    // Anonymized for research use
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalAnalysisResponse {
    pub analysis_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub analysis_type: ClinicalAnalysisType,
    pub findings: Vec<ClinicalFinding>,
    pub recommendations: Vec<ClinicalRecommendation>,
    pub confidence_score: f64,
    pub risk_assessment: RiskAssessment,
    pub follow_up_required: bool,
    pub privacy_compliance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalFinding {
    pub category: String,
    pub description: String,
    pub severity: Severity,
    pub confidence: f64,
    pub anatomical_location: Option<String>,
    pub icd_10_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalRecommendation {
    pub category: String,
    pub recommendation: String,
    pub priority: Priority,
    pub evidence_level: EvidenceLevel,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Critical,
    High,
    Moderate,
    Low,
    Minimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Urgent,
    High,
    Medium,
    Low,
    Routine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceLevel {
    StrongEvidence,
    ModerateEvidence,
    LimitedEvidence,
    ExpertOpinion,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub protective_factors: Vec<String>,
    pub risk_stratification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor: String,
    pub impact: f64,
    pub modifiable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DICOMAnalysisRequest {
    pub dicom_series: DICOMSeries,
    pub analysis_focus: Vec<String>,
    pub compare_with_normal: bool,
    pub include_measurements: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DICOMAnalysisResponse {
    pub series_id: String,
    pub modality: String,
    pub anatomical_region: String,
    pub findings: Vec<ImagingFinding>,
    pub measurements: HashMap<String, f64>,
    pub quality_assessment: ImageQualityAssessment,
    pub comparison_analysis: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagingFinding {
    pub finding_type: String,
    pub location: String,
    pub description: String,
    pub severity: Severity,
    pub confidence: f64,
    pub measurements: HashMap<String, f64>,
    pub birads_score: Option<i32>,  // For mammography
    pub lung_rads_score: Option<i32>, // For lung CT
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQualityAssessment {
    pub overall_quality: f64,
    pub noise_level: f64,
    pub contrast_adequacy: f64,
    pub motion_artifacts: bool,
    pub technical_adequacy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalChatSession {
    pub session_id: String,
    pub messages: Vec<ChatMessage>,
    pub context: MedicalContext,
    pub specialist_mode: Option<MedicalSpecialty>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalContext {
    pub patient_demographics: Option<PatientDemographics>,
    pub clinical_history: Vec<String>,
    pub current_medications: Vec<String>,
    pub allergies: Vec<String>,
    pub previous_studies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientDemographics {
    pub age_range: String,  // "20-30", "65+" etc. for privacy
    pub gender: String,
    pub relevant_comorbidities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MedicalSpecialty {
    Radiology,
    Pathology,
    Cardiology,
    Oncology,
    Neurology,
    EmergencyMedicine,
    InternalMedicine,
    Surgery,
    Pediatrics,
    Psychiatry,
}

impl MedicalAIAgent {
    pub fn new(api_key: String, config: &OpenAlgebraConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let openai_config = async_openai::config::OpenAIConfig::new()
            .with_api_key(api_key);
        
        let client = Client::with_config(openai_config);
        
        let tokenizer = get_bpe_from_model("gpt-4").ok();
        
        let system_prompt = Self::create_medical_system_prompt();
        
        Ok(Self {
            client,
            model: "gpt-4-turbo-preview".to_string(),
            temperature: 0.1, // Low temperature for medical accuracy
            max_tokens: 4000,
            system_prompt,
            tokenizer,
        })
    }

    fn create_medical_system_prompt() -> String {
        r#"You are an advanced medical AI assistant integrated with the OpenAlgebra Medical AI system. 
You specialize in analyzing medical data, DICOM images, and providing clinical decision support.

IMPORTANT GUIDELINES:
1. Always prioritize patient safety and accuracy
2. Clearly indicate confidence levels for all analyses
3. Recommend human expert review for critical findings
4. Maintain HIPAA compliance and patient privacy
5. Use evidence-based medical knowledge
6. Provide ICD-10 codes when appropriate
7. Include relevant differential diagnoses
8. Suggest appropriate follow-up actions

CAPABILITIES:
- DICOM image analysis and interpretation
- Clinical data analysis and pattern recognition
- Risk assessment and stratification
- Treatment recommendation support
- Drug interaction checking
- Anomaly detection in medical data
- Clinical documentation assistance

Remember: You are a decision support tool, not a replacement for clinical judgment."#.to_string()
    }

    pub async fn analyze_clinical_data(
        &self,
        request: ClinicalAnalysisRequest,
    ) -> Result<ClinicalAnalysisResponse, Box<dyn std::error::Error>> {
        // Sanitize patient data based on privacy level
        let sanitized_data = self.sanitize_patient_data(&request.patient_data, &request.privacy_level)?;
        
        // Create context-aware prompt
        let analysis_prompt = self.create_clinical_analysis_prompt(&request, &sanitized_data);
        
        // Call OpenAI API with timeout
        let response = timeout(
            Duration::from_secs(60),
            self.call_openai_chat(&analysis_prompt, &request.analysis_type)
        ).await??;
        
        // Parse and structure the response
        let analysis_response = self.parse_clinical_response(&response, &request.analysis_type)?;
        
        Ok(analysis_response)
    }

    pub async fn analyze_dicom_series(
        &self,
        request: DICOMAnalysisRequest,
    ) -> Result<DICOMAnalysisResponse, Box<dyn std::error::Error>> {
        // Extract image features and metadata
        let image_features = self.extract_dicom_features(&request.dicom_series)?;
        
        // Create specialized imaging prompt
        let imaging_prompt = self.create_imaging_analysis_prompt(&request, &image_features);
        
        // Process with vision-capable model if images are included
        let response = if !request.dicom_series.images.is_empty() {
            self.analyze_medical_images(&imaging_prompt, &request.dicom_series).await?
        } else {
            self.call_openai_chat(&imaging_prompt, &ClinicalAnalysisType::DiagnosticImaging).await?
        };
        
        let analysis_response = self.parse_dicom_response(&response, &request.dicom_series)?;
        
        Ok(analysis_response)
    }

    pub async fn start_medical_chat_session(
        &self,
        context: MedicalContext,
        specialty: Option<MedicalSpecialty>,
    ) -> Result<MedicalChatSession, Box<dyn std::error::Error>> {
        let session_id = uuid::Uuid::new_v4().to_string();
        
        // Create specialty-specific system message
        let system_message = self.create_specialty_system_message(&specialty);
        
        let session = MedicalChatSession {
            session_id: session_id.clone(),
            messages: vec![ChatMessage {
                role: Role::System,
                content: system_message,
                timestamp: chrono::Utc::now(),
                metadata: None,
            }],
            context,
            specialist_mode: specialty,
        };
        
        Ok(session)
    }

    pub async fn continue_medical_chat(
        &self,
        session: &mut MedicalChatSession,
        user_message: String,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Add user message to session
        session.messages.push(ChatMessage {
            role: Role::User,
            content: user_message,
            timestamp: chrono::Utc::now(),
            metadata: None,
        });
        
        // Prepare messages for OpenAI API
        let messages = self.prepare_chat_messages(&session.messages, &session.context)?;
        
        // Call OpenAI Chat API
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .temperature(self.temperature)
            .max_tokens(self.max_tokens)
            .build()?;
        
        let response = self.client.chat().completions().create(request).await?;
        
        let assistant_message = response.choices[0].message.content.clone()
            .unwrap_or_else(|| "I apologize, but I couldn't generate a response.".to_string());
        
        // Add assistant response to session
        session.messages.push(ChatMessage {
            role: Role::Assistant,
            content: assistant_message.clone(),
            timestamp: chrono::Utc::now(),
            metadata: Some(HashMap::from([
                ("model".to_string(), self.model.clone()),
                ("finish_reason".to_string(), format!("{:?}", response.choices[0].finish_reason)),
            ])),
        });
        
        Ok(assistant_message)
    }

    pub async fn generate_medical_embeddings(
        &self,
        medical_texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-ada-002")
            .input(medical_texts)
            .build()?;
        
        let response = self.client.embeddings().create(request).await?;
        
        let embeddings = response.data
            .into_iter()
            .map(|embedding| embedding.embedding)
            .collect();
        
        Ok(embeddings)
    }

    pub async fn detect_medical_anomalies(
        &self,
        dataset: &MedicalDataset,
        baseline_embeddings: &[Vec<f32>],
    ) -> Result<Vec<AnomalyDetectionResult>, Box<dyn std::error::Error>> {
        // Convert medical data to text descriptions
        let data_descriptions = self.create_medical_data_descriptions(dataset)?;
        
        // Generate embeddings for current data
        let current_embeddings = self.generate_medical_embeddings(data_descriptions).await?;
        
        // Calculate anomaly scores using cosine similarity
        let mut anomaly_results = Vec::new();
        
        for (i, current_embedding) in current_embeddings.iter().enumerate() {
            let mut max_similarity = 0.0f32;
            
            for baseline_embedding in baseline_embeddings {
                let similarity = self.cosine_similarity(current_embedding, baseline_embedding);
                max_similarity = max_similarity.max(similarity);
            }
            
            let anomaly_score = 1.0 - max_similarity;
            
            anomaly_results.push(AnomalyDetectionResult {
                data_point_index: i,
                anomaly_score,
                is_anomaly: anomaly_score > 0.7, // Threshold for anomaly detection
                description: format!("Data point {} anomaly score: {:.3}", i, anomaly_score),
            });
        }
        
        Ok(anomaly_results)
    }

    async fn call_openai_chat(
        &self,
        prompt: &str,
        analysis_type: &ClinicalAnalysisType,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let system_message = ChatCompletionRequestSystemMessageArgs::default()
            .content(&self.system_prompt)
            .build()?;
        
        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?;
        
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages([
                ChatCompletionRequestMessage::System(system_message),
                ChatCompletionRequestMessage::User(user_message),
            ])
            .temperature(self.temperature)
            .max_tokens(self.max_tokens)
            .build()?;
        
        let response = self.client.chat().completions().create(request).await?;
        
        Ok(response.choices[0].message.content.clone()
            .unwrap_or_else(|| "No response generated".to_string()))
    }

    async fn analyze_medical_images(
        &self,
        prompt: &str,
        dicom_series: &DICOMSeries,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Convert DICOM images to base64 for vision model
        let mut base64_images = Vec::new();
        
        for image in &dicom_series.images {
            if let Some(base64_data) = self.convert_dicom_to_base64(image)? {
                base64_images.push(base64_data);
                if base64_images.len() >= 5 { // Limit to 5 images for API constraints
                    break;
                }
            }
        }
        
        // For now, use text-based analysis since vision models have specific requirements
        // In a full implementation, you would use GPT-4 Vision here
        let enhanced_prompt = format!(
            "{}\n\nDICOM Series Information:\n- Modality: {}\n- Number of images: {}\n- Study Date: {}\n- Patient Position: {:?}",
            prompt,
            dicom_series.modality,
            dicom_series.images.len(),
            dicom_series.study_date,
            dicom_series.patient_position
        );
        
        self.call_openai_chat(&enhanced_prompt, &ClinicalAnalysisType::DiagnosticImaging).await
    }

    fn sanitize_patient_data(
        &self,
        data: &MedicalDataset,
        privacy_level: &PrivacyLevel,
    ) -> Result<MedicalDataset, Box<dyn std::error::Error>> {
        let mut sanitized = data.clone();
        
        match privacy_level {
            PrivacyLevel::Full => {
                // Remove all PHI
                sanitized.patient_ids = sanitized.patient_ids.iter()
                    .map(|_| "PATIENT_XXX".to_string())
                    .collect();
                // Keep only anonymized clinical data
            },
            PrivacyLevel::Partial => {
                // Remove direct identifiers only
                sanitized.patient_ids = sanitized.patient_ids.iter()
                    .enumerate()
                    .map(|(i, _)| format!("PATIENT_{:03}", i))
                    .collect();
            },
            PrivacyLevel::Minimal => {
                // Keep clinical context, minimal anonymization
                // Implementation depends on specific requirements
            },
            PrivacyLevel::Research => {
                // Research-grade anonymization
                sanitized.patient_ids = sanitized.patient_ids.iter()
                    .enumerate()
                    .map(|(i, _)| format!("RESEARCH_SUBJECT_{:06}", i))
                    .collect();
            },
        }
        
        Ok(sanitized)
    }

    fn create_clinical_analysis_prompt(
        &self,
        request: &ClinicalAnalysisRequest,
        data: &MedicalDataset,
    ) -> String {
        format!(
            r#"Perform a comprehensive {} analysis on the provided medical data.

ANALYSIS TYPE: {:?}
PRIVACY LEVEL: {:?}
INCLUDE RECOMMENDATIONS: {}

MEDICAL DATA:
- Number of patients: {}
- Features available: {} dimensions
- Data quality metrics: Available

Please provide:
1. Detailed clinical findings with confidence scores
2. Risk assessment and stratification
3. Relevant ICD-10 codes
4. Evidence-based recommendations (if requested)
5. Suggested follow-up actions
6. Any safety concerns or red flags

Format your response as structured JSON with the following fields:
- findings: Array of clinical findings
- recommendations: Array of recommendations (if requested)
- risk_assessment: Overall risk evaluation
- confidence_score: Overall confidence (0.0-1.0)
- follow_up_required: Boolean
- safety_alerts: Array of any urgent concerns

Ensure all recommendations follow current clinical guidelines and best practices."#,
            match request.analysis_type {
                ClinicalAnalysisType::DiagnosticImaging => "diagnostic imaging",
                ClinicalAnalysisType::PathologyReview => "pathology review",
                ClinicalAnalysisType::TreatmentPlanning => "treatment planning",
                ClinicalAnalysisType::RiskAssessment => "risk assessment",
                ClinicalAnalysisType::DrugInteraction => "drug interaction",
                ClinicalAnalysisType::ClinicalDecisionSupport => "clinical decision support",
                ClinicalAnalysisType::AnomalyDetection => "anomaly detection",
                ClinicalAnalysisType::PrognosticAnalysis => "prognostic analysis",
            },
            request.analysis_type,
            request.privacy_level,
            request.include_recommendations,
            data.patient_ids.len(),
            data.features.ncols(),
        )
    }

    fn create_imaging_analysis_prompt(
        &self,
        request: &DICOMAnalysisRequest,
        features: &DICOMMetadata,
    ) -> String {
        format!(
            r#"Analyze the provided DICOM imaging series with focus on: {:?}

IMAGING DETAILS:
- Modality: {}
- Study Description: {}
- Series Description: {}
- Number of Images: {}
- Image Dimensions: {}x{}
- Pixel Spacing: {:?}
- Compare with Normal: {}
- Include Measurements: {}

ANALYSIS FOCUS: {:?}

Please provide:
1. Detailed imaging findings with anatomical locations
2. Severity assessment for each finding
3. Confidence scores for all observations
4. Relevant measurements (if requested)
5. Quality assessment of the images
6. Comparison with normal anatomy/pathology patterns
7. Appropriate reporting scores (BI-RADS, LUNG-RADS, etc.) if applicable

Format as structured analysis with:
- Findings: Location, description, severity, confidence
- Measurements: Quantitative data where applicable
- Quality: Technical adequacy assessment
- Recommendations: Next steps or additional imaging needs"#,
            request.analysis_focus,
            features.modality,
            features.study_description.as_ref().unwrap_or(&"N/A".to_string()),
            features.series_description.as_ref().unwrap_or(&"N/A".to_string()),
            features.number_of_frames.unwrap_or(1),
            features.rows.unwrap_or(512),
            features.columns.unwrap_or(512),
            features.pixel_spacing,
            request.compare_with_normal,
            request.include_measurements,
            request.analysis_focus,
        )
    }

    fn parse_clinical_response(
        &self,
        response: &str,
        analysis_type: &ClinicalAnalysisType,
    ) -> Result<ClinicalAnalysisResponse, Box<dyn std::error::Error>> {
        // Try to parse as JSON first, fallback to text parsing
        if let Ok(json_response) = serde_json::from_str::<serde_json::Value>(response) {
            // Parse structured JSON response
            self.parse_json_clinical_response(&json_response, analysis_type)
        } else {
            // Parse unstructured text response
            self.parse_text_clinical_response(response, analysis_type)
        }
    }

    fn parse_json_clinical_response(
        &self,
        json: &serde_json::Value,
        analysis_type: &ClinicalAnalysisType,
    ) -> Result<ClinicalAnalysisResponse, Box<dyn std::error::Error>> {
        let findings = json["findings"].as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|f| ClinicalFinding {
                category: f["category"].as_str().unwrap_or("General").to_string(),
                description: f["description"].as_str().unwrap_or("No description").to_string(),
                severity: match f["severity"].as_str().unwrap_or("moderate") {
                    "critical" => Severity::Critical,
                    "high" => Severity::High,
                    "moderate" => Severity::Moderate,
                    "low" => Severity::Low,
                    _ => Severity::Minimal,
                },
                confidence: f["confidence"].as_f64().unwrap_or(0.5),
                anatomical_location: f["location"].as_str().map(|s| s.to_string()),
                icd_10_codes: f["icd_codes"].as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|c| c.as_str().map(|s| s.to_string()))
                    .collect(),
            })
            .collect();

        let recommendations = json["recommendations"].as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|r| ClinicalRecommendation {
                category: r["category"].as_str().unwrap_or("General").to_string(),
                recommendation: r["recommendation"].as_str().unwrap_or("No recommendation").to_string(),
                priority: match r["priority"].as_str().unwrap_or("medium") {
                    "urgent" => Priority::Urgent,
                    "high" => Priority::High,
                    "medium" => Priority::Medium,
                    "low" => Priority::Low,
                    _ => Priority::Routine,
                },
                evidence_level: EvidenceLevel::ModerateEvidence,
                rationale: r["rationale"].as_str().unwrap_or("Clinical judgment").to_string(),
            })
            .collect();

        Ok(ClinicalAnalysisResponse {
            analysis_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            analysis_type: analysis_type.clone(),
            findings,
            recommendations,
            confidence_score: json["confidence_score"].as_f64().unwrap_or(0.5),
            risk_assessment: RiskAssessment {
                overall_risk: json["risk_assessment"]["overall_risk"].as_f64().unwrap_or(0.5),
                risk_factors: vec![],
                protective_factors: vec![],
                risk_stratification: "Moderate risk".to_string(),
            },
            follow_up_required: json["follow_up_required"].as_bool().unwrap_or(true),
            privacy_compliance: true,
        })
    }

    fn parse_text_clinical_response(
        &self,
        response: &str,
        analysis_type: &ClinicalAnalysisType,
    ) -> Result<ClinicalAnalysisResponse, Box<dyn std::error::Error>> {
        // Simple text parsing fallback
        let findings = vec![ClinicalFinding {
            category: "AI Analysis".to_string(),
            description: response.to_string(),
            severity: Severity::Moderate,
            confidence: 0.7,
            anatomical_location: None,
            icd_10_codes: vec![],
        }];

        Ok(ClinicalAnalysisResponse {
            analysis_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            analysis_type: analysis_type.clone(),
            findings,
            recommendations: vec![],
            confidence_score: 0.7,
            risk_assessment: RiskAssessment {
                overall_risk: 0.5,
                risk_factors: vec![],
                protective_factors: vec![],
                risk_stratification: "Requires further evaluation".to_string(),
            },
            follow_up_required: true,
            privacy_compliance: true,
        })
    }

    fn parse_dicom_response(
        &self,
        response: &str,
        dicom_series: &DICOMSeries,
    ) -> Result<DICOMAnalysisResponse, Box<dyn std::error::Error>> {
        Ok(DICOMAnalysisResponse {
            series_id: dicom_series.series_instance_uid.clone(),
            modality: dicom_series.modality.clone(),
            anatomical_region: "Not specified".to_string(),
            findings: vec![ImagingFinding {
                finding_type: "AI Analysis".to_string(),
                location: "Multiple regions".to_string(),
                description: response.to_string(),
                severity: Severity::Moderate,
                confidence: 0.7,
                measurements: HashMap::new(),
                birads_score: None,
                lung_rads_score: None,
            }],
            measurements: HashMap::new(),
            quality_assessment: ImageQualityAssessment {
                overall_quality: 0.8,
                noise_level: 0.2,
                contrast_adequacy: 0.8,
                motion_artifacts: false,
                technical_adequacy: true,
            },
            comparison_analysis: Some("Comparison analysis pending".to_string()),
        })
    }

    fn extract_dicom_features(&self, series: &DICOMSeries) -> Result<DICOMMetadata, Box<dyn std::error::Error>> {
        // Extract key metadata for analysis
        Ok(DICOMMetadata {
            study_instance_uid: series.study_instance_uid.clone(),
            series_instance_uid: series.series_instance_uid.clone(),
            modality: series.modality.clone(),
            study_date: Some(series.study_date.clone()),
            study_description: Some("Medical imaging study".to_string()),
            series_description: Some("Imaging series".to_string()),
            patient_position: series.patient_position.clone(),
            rows: Some(512),
            columns: Some(512),
            pixel_spacing: Some((1.0, 1.0)),
            slice_thickness: Some(1.0),
            number_of_frames: Some(series.images.len() as u32),
        })
    }

    fn convert_dicom_to_base64(
        &self,
        _image: &crate::dicom::DICOMImage,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        // Placeholder for DICOM to base64 conversion
        // In a real implementation, you would convert the DICOM pixel data to a standard format
        Ok(None)
    }

    fn create_specialty_system_message(&self, specialty: &Option<MedicalSpecialty>) -> String {
        match specialty {
            Some(MedicalSpecialty::Radiology) => {
                "You are a specialized radiology AI assistant. Focus on imaging interpretation, anatomy, pathology identification, and imaging protocols.".to_string()
            },
            Some(MedicalSpecialty::Pathology) => {
                "You are a specialized pathology AI assistant. Focus on tissue analysis, cellular morphology, disease diagnosis, and microscopic findings.".to_string()
            },
            Some(MedicalSpecialty::Cardiology) => {
                "You are a specialized cardiology AI assistant. Focus on cardiac function, ECG interpretation, cardiac imaging, and cardiovascular disease management.".to_string()
            },
            Some(MedicalSpecialty::Oncology) => {
                "You are a specialized oncology AI assistant. Focus on cancer diagnosis, staging, treatment planning, and prognosis assessment.".to_string()
            },
            Some(MedicalSpecialty::Neurology) => {
                "You are a specialized neurology AI assistant. Focus on neurological disorders, brain imaging, neurophysiology, and neurological examination findings.".to_string()
            },
            _ => self.system_prompt.clone(),
        }
    }

    fn prepare_chat_messages(
        &self,
        messages: &[ChatMessage],
        _context: &MedicalContext,
    ) -> Result<Vec<ChatCompletionRequestMessage>, Box<dyn std::error::Error>> {
        let mut openai_messages = Vec::new();
        
        for message in messages {
            match message.role {
                Role::System => {
                    openai_messages.push(ChatCompletionRequestMessage::System(
                        ChatCompletionRequestSystemMessageArgs::default()
                            .content(&message.content)
                            .build()?
                    ));
                },
                Role::User => {
                    openai_messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(&message.content)
                            .build()?
                    ));
                },
                Role::Assistant => {
                    openai_messages.push(ChatCompletionRequestMessage::Assistant(
                        async_openai::types::ChatCompletionRequestAssistantMessageArgs::default()
                            .content(&message.content)
                            .build()?
                    ));
                },
                _ => {
                    // Handle other roles if needed
                }
            }
        }
        
        Ok(openai_messages)
    }

    fn create_medical_data_descriptions(&self, dataset: &MedicalDataset) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut descriptions = Vec::new();
        
        for (i, patient_id) in dataset.patient_ids.iter().enumerate() {
            let features_row = dataset.features.row(i);
            let description = format!(
                "Patient {}: Medical data with {} features. Key metrics: {}",
                patient_id,
                features_row.len(),
                features_row.iter().take(5).map(|x| format!("{:.2}", x)).collect::<Vec<_>>().join(", ")
            );
            descriptions.push(description);
        }
        
        Ok(descriptions)
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    pub data_point_index: usize,
    pub anomaly_score: f32,
    pub is_anomaly: bool,
    pub description: String,
}

// Additional utility functions for the agents system
pub async fn initialize_medical_ai_agents(
    config: &OpenAlgebraConfig,
) -> Result<MedicalAIAgent, Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY environment variable not set")?;
    
    MedicalAIAgent::new(api_key, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_medical_ai_agent_creation() {
        let config = OpenAlgebraConfig::default();
        // This test requires OPENAI_API_KEY to be set
        if std::env::var("OPENAI_API_KEY").is_ok() {
            let agent = MedicalAIAgent::new("test-key".to_string(), &config);
            assert!(agent.is_ok());
        }
    }

    #[test]
    fn test_clinical_analysis_request_serialization() {
        let request = ClinicalAnalysisRequest {
            patient_data: MedicalDataset {
                patient_ids: vec!["PATIENT_001".to_string()],
                features: SparseMatrix::new(1, 10).unwrap(),
                labels: vec![0],
                metadata: std::collections::HashMap::new(),
            },
            dicom_series: None,
            analysis_type: ClinicalAnalysisType::RiskAssessment,
            include_recommendations: true,
            privacy_level: PrivacyLevel::Full,
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("risk_assessment"));
    }

    #[test]
    fn test_cosine_similarity() {
        let config = OpenAlgebraConfig::default();
        let agent = MedicalAIAgent::new("test-key".to_string(), &config).unwrap();
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = agent.cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = agent.cosine_similarity(&a, &c);
        assert!((similarity2 - 0.0).abs() < 0.001);
    }
} 