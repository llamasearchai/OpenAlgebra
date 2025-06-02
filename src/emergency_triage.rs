//! Emergency Medicine Triage AI System


//! 
//! Rapid patient assessment and prioritization for emergency departments
//! with life-saving decision support and resource optimization.

use crate::{
    api::ApiError,
    models::{MedicalData, ProcessingResult},
    sparse::SparseMatrix,
    medical::MedicalProcessor,
    agents::MedicalAgent,
};
use std::collections::{HashMap, BinaryHeap, VecDeque};
use std::cmp::Ordering;
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyPatient {
    pub patient_id: String,
    pub arrival_time: DateTime<Utc>,
    pub chief_complaint: String,
    pub vital_signs: VitalSigns,
    pub symptoms: Vec<Symptom>,
    pub medical_history: MedicalHistory,
    pub initial_assessment: InitialAssessment,
    pub triage_score: Option<TriageScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitalSigns {
    pub heart_rate: f32,
    pub blood_pressure_systolic: f32,
    pub blood_pressure_diastolic: f32,
    pub respiratory_rate: f32,
    pub temperature_celsius: f32,
    pub oxygen_saturation: f32,
    pub pain_scale: u8,
    pub glasgow_coma_scale: u8,
    pub blood_glucose: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symptom {
    pub name: String,
    pub severity: SeverityLevel,
    pub duration_minutes: u32,
    pub onset: SymptomOnset,
    pub associated_symptoms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SeverityLevel {
    Mild = 1,
    Moderate = 2,
    Severe = 3,
    Critical = 4,
    LifeThreatening = 5,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymptomOnset {
    Sudden,
    Gradual,
    Intermittent,
    Chronic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalHistory {
    pub chronic_conditions: Vec<String>,
    pub medications: Vec<Medication>,
    pub allergies: Vec<Allergy>,
    pub recent_surgeries: Vec<Surgery>,
    pub immunization_status: HashMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Medication {
    pub name: String,
    pub dosage: String,
    pub frequency: String,
    pub start_date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allergy {
    pub allergen: String,
    pub reaction: String,
    pub severity: SeverityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surgery {
    pub procedure: String,
    pub date: DateTime<Utc>,
    pub complications: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialAssessment {
    pub performed_by: String,
    pub assessment_time: DateTime<Utc>,
    pub observations: Vec<String>,
    pub diagnostic_tests_ordered: Vec<DiagnosticTest>,
    pub immediate_interventions: Vec<Intervention>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticTest {
    pub test_type: String,
    pub priority: TestPriority,
    pub reason: String,
    pub results: Option<TestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestPriority {
    Routine = 1,
    Urgent = 2,
    Stat = 3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub value: String,
    pub unit: String,
    pub normal_range: String,
    pub abnormal: bool,
    pub critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intervention {
    pub intervention_type: String,
    pub timestamp: DateTime<Utc>,
    pub performed_by: String,
    pub outcome: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageScore {
    pub category: TriageCategory,
    pub priority_score: f32,
    pub estimated_wait_time: Duration,
    pub recommended_actions: Vec<RecommendedAction>,
    pub risk_factors: Vec<RiskFactor>,
    pub ai_confidence: f32,
    pub explanation: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TriageCategory {
    Resuscitation = 1,      // Immediate life-saving intervention required
    Emergency = 2,          // Emergent, life-threatening condition
    Urgent = 3,            // Urgent, potentially life-threatening
    SemiUrgent = 4,        // Semi-urgent, situational urgency
    NonUrgent = 5,         // Non-urgent, can be delayed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedAction {
    pub action: String,
    pub priority: ActionPriority,
    pub responsible_party: String,
    pub time_constraint: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor: String,
    pub risk_level: RiskLevel,
    pub mitigation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low = 1,
    Moderate = 2,
    High = 3,
    Critical = 4,
}

impl Eq for EmergencyPatient {}

impl PartialEq for EmergencyPatient {
    fn eq(&self, other: &Self) -> bool {
        self.patient_id == other.patient_id
    }
}

impl Ord for EmergencyPatient {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority patients come first
        match (&self.triage_score, &other.triage_score) {
            (Some(a), Some(b)) => b.category.cmp(&a.category)
                .then_with(|| b.priority_score.partial_cmp(&a.priority_score).unwrap_or(Ordering::Equal))
                .then_with(|| a.arrival_time.cmp(&b.arrival_time)),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.arrival_time.cmp(&b.arrival_time),
        }
    }
}

impl PartialOrd for EmergencyPatient {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct EmergencyTriageSystem {
    patient_queue: Arc<Mutex<BinaryHeap<EmergencyPatient>>>,
    active_patients: Arc<RwLock<HashMap<String, EmergencyPatient>>>,
    medical_agent: Arc<MedicalAgent>,
    resource_manager: Arc<RwLock<ResourceManager>>,
    prediction_models: Arc<RwLock<PredictionModels>>,
    alert_system: Arc<Mutex<AlertSystem>>,
}

struct ResourceManager {
    available_beds: HashMap<String, u32>,
    staff_availability: HashMap<String, Vec<StaffMember>>,
    equipment_status: HashMap<String, EquipmentStatus>,
    current_wait_times: HashMap<TriageCategory, Duration>,
}

struct StaffMember {
    id: String,
    role: String,
    specializations: Vec<String>,
    current_patient: Option<String>,
    available: bool,
}

#[derive(Debug, Clone)]
struct EquipmentStatus {
    equipment_type: String,
    total_units: u32,
    available_units: u32,
    maintenance_required: Vec<String>,
}

struct PredictionModels {
    deterioration_model: SparseMatrix,
    diagnosis_model: SparseMatrix,
    resource_prediction_model: SparseMatrix,
    outcome_model: SparseMatrix,
}

struct AlertSystem {
    active_alerts: Vec<Alert>,
    alert_history: VecDeque<Alert>,
    notification_channels: HashMap<String, NotificationChannel>,
}

#[derive(Debug, Clone)]
struct Alert {
    alert_id: String,
    alert_type: AlertType,
    patient_id: String,
    severity: SeverityLevel,
    message: String,
    timestamp: DateTime<Utc>,
    acknowledged: bool,
}

#[derive(Debug, Clone)]
enum AlertType {
    PatientDeterioration,
    CriticalLabResult,
    ResourceShortage,
    WaitTimeExceeded,
    SystemOverload,
}

#[derive(Debug, Clone)]
struct NotificationChannel {
    channel_type: String,
    recipients: Vec<String>,
    active: bool,
}

use std::sync::Arc;

impl EmergencyTriageSystem {
    pub async fn new(medical_agent: MedicalAgent) -> Result<Self, ApiError> {
        let resource_manager = ResourceManager {
            available_beds: Self::initialize_bed_availability(),
            staff_availability: Self::initialize_staff(),
            equipment_status: Self::initialize_equipment(),
            current_wait_times: Self::initialize_wait_times(),
        };
        
        let prediction_models = PredictionModels {
            deterioration_model: Self::load_deterioration_model()?,
            diagnosis_model: Self::load_diagnosis_model()?,
            resource_prediction_model: Self::load_resource_model()?,
            outcome_model: Self::load_outcome_model()?,
        };
        
        let alert_system = AlertSystem {
            active_alerts: Vec::new(),
            alert_history: VecDeque::with_capacity(1000),
            notification_channels: Self::initialize_notification_channels(),
        };
        
        Ok(Self {
            patient_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            active_patients: Arc::new(RwLock::new(HashMap::new())),
            medical_agent: Arc::new(medical_agent),
            resource_manager: Arc::new(RwLock::new(resource_manager)),
            prediction_models: Arc::new(RwLock::new(prediction_models)),
            alert_system: Arc::new(Mutex::new(alert_system)),
        })
    }
    
    pub async fn triage_patient(&self, mut patient: EmergencyPatient) -> Result<TriageScore, ApiError> {
        // Perform AI-driven triage assessment
        let vital_analysis = self.analyze_vital_signs(&patient.vital_signs).await?;
        let symptom_analysis = self.analyze_symptoms(&patient.symptoms).await?;
        let history_risk = self.assess_medical_history_risk(&patient.medical_history).await?;
        
        // Use medical agent for advanced analysis
        let ai_assessment = self.medical_agent.assess_emergency_patient(
            &patient,
            &vital_analysis,
            &symptom_analysis,
            history_risk,
        ).await?;
        
        // Calculate triage score
        let triage_score = self.calculate_triage_score(
            vital_analysis,
            symptom_analysis,
            history_risk,
            ai_assessment,
        ).await?;
        
        // Update patient with triage score
        patient.triage_score = Some(triage_score.clone());
        
        // Add to queue and active patients
        let mut queue = self.patient_queue.lock().await;
        queue.push(patient.clone());
        
        let mut active = self.active_patients.write().await;
        active.insert(patient.patient_id.clone(), patient.clone());
        
        // Check for alerts
        self.check_and_generate_alerts(&patient, &triage_score).await?;
        
        // Update resource predictions
        self.update_resource_predictions().await?;
        
        Ok(triage_score)
    }
    
    async fn analyze_vital_signs(&self, vitals: &VitalSigns) -> Result<VitalAnalysis, ApiError> {
        let mut critical_findings = Vec::new();
        let mut risk_score = 0.0;
        
        // Heart rate analysis
        if vitals.heart_rate < 40.0 || vitals.heart_rate > 150.0 {
            critical_findings.push("Critical heart rate".to_string());
            risk_score += 3.0;
        } else if vitals.heart_rate < 50.0 || vitals.heart_rate > 120.0 {
            critical_findings.push("Abnormal heart rate".to_string());
            risk_score += 1.5;
        }
        
        // Blood pressure analysis
        if vitals.blood_pressure_systolic < 90.0 || vitals.blood_pressure_systolic > 200.0 {
            critical_findings.push("Critical blood pressure".to_string());
            risk_score += 3.0;
        }
        
        // Respiratory rate analysis
        if vitals.respiratory_rate < 8.0 || vitals.respiratory_rate > 30.0 {
            critical_findings.push("Critical respiratory rate".to_string());
            risk_score += 3.0;
        }
        
        // Oxygen saturation analysis
        if vitals.oxygen_saturation < 90.0 {
            critical_findings.push("Critical oxygen saturation".to_string());
            risk_score += 3.0;
        } else if vitals.oxygen_saturation < 95.0 {
            critical_findings.push("Low oxygen saturation".to_string());
            risk_score += 1.0;
        }
        
        // Temperature analysis
        if vitals.temperature_celsius < 35.0 || vitals.temperature_celsius > 40.0 {
            critical_findings.push("Critical temperature".to_string());
            risk_score += 2.0;
        }
        
        // Glasgow Coma Scale
        if vitals.glasgow_coma_scale < 9 {
            critical_findings.push("Severe altered consciousness".to_string());
            risk_score += 4.0;
        } else if vitals.glasgow_coma_scale < 13 {
            critical_findings.push("Moderate altered consciousness".to_string());
            risk_score += 2.0;
        }
        
        Ok(VitalAnalysis {
            critical_findings,
            risk_score,
            requires_immediate_intervention: risk_score >= 3.0,
        })
    }
    
    async fn analyze_symptoms(&self, symptoms: &[Symptom]) -> Result<SymptomAnalysis, ApiError> {
        let mut severity_score = 0.0;
        let mut critical_symptoms = Vec::new();
        let mut differential_diagnoses = Vec::new();
        
        for symptom in symptoms {
            match symptom.severity {
                SeverityLevel::LifeThreatening => {
                    severity_score += 5.0;
                    critical_symptoms.push(symptom.name.clone());
                }
                SeverityLevel::Critical => {
                    severity_score += 4.0;
                    critical_symptoms.push(symptom.name.clone());
                }
                SeverityLevel::Severe => severity_score += 3.0,
                SeverityLevel::Moderate => severity_score += 2.0,
                SeverityLevel::Mild => severity_score += 1.0,
            }
            
            // Check for red flag symptoms
            if self.is_red_flag_symptom(&symptom.name) {
                severity_score += 2.0;
                critical_symptoms.push(format!("RED FLAG: {}", symptom.name));
            }
        }
        
        // Generate differential diagnoses using AI
        let models = self.prediction_models.read().await;
        differential_diagnoses = self.generate_differential_diagnoses(symptoms, &models.diagnosis_model)?;
        
        Ok(SymptomAnalysis {
            severity_score,
            critical_symptoms,
            differential_diagnoses,
            recommended_tests: self.recommend_diagnostic_tests(&differential_diagnoses),
        })
    }
    
    async fn assess_medical_history_risk(&self, history: &MedicalHistory) -> Result<f32, ApiError> {
        let mut risk_score = 0.0;
        
        // Chronic conditions risk
        for condition in &history.chronic_conditions {
            risk_score += match condition.as_str() {
                "Diabetes" => 1.5,
                "Heart Disease" => 2.0,
                "COPD" => 1.8,
                "Cancer" => 2.5,
                "Immunocompromised" => 2.0,
                _ => 0.5,
            };
        }
        
        // Medication interactions risk
        if history.medications.len() > 5 {
            risk_score += 1.0;
        }
        
        // Allergy risk
        for allergy in &history.allergies {
            if allergy.severity >= SeverityLevel::Severe {
                risk_score += 1.5;
            }
        }
        
        // Recent surgery risk
        for surgery in &history.recent_surgeries {
            let days_since = (Utc::now() - surgery.date).num_days();
            if days_since < 30 {
                risk_score += 2.0;
            } else if days_since < 90 {
                risk_score += 1.0;
            }
        }
        
        Ok(risk_score)
    }
    
    fn calculate_triage_score(
        &self,
        vital_analysis: VitalAnalysis,
        symptom_analysis: SymptomAnalysis,
        history_risk: f32,
        ai_assessment: crate::agents::EmergencyAssessment,
    ) -> impl std::future::Future<Output = Result<TriageScore, ApiError>> {
        async move {
            // Combine all factors
            let total_score = vital_analysis.risk_score + 
                             symptom_analysis.severity_score + 
                             history_risk +
                             ai_assessment.urgency_score;
            
            // Determine triage category
            let category = if vital_analysis.requires_immediate_intervention || 
                             ai_assessment.immediate_intervention_required {
                TriageCategory::Resuscitation
            } else if total_score > 10.0 {
                TriageCategory::Emergency
            } else if total_score > 6.0 {
                TriageCategory::Urgent
            } else if total_score > 3.0 {
                TriageCategory::SemiUrgent
            } else {
                TriageCategory::NonUrgent
            };
            
            // Estimate wait time based on current ED status
            let wait_time = match category {
                TriageCategory::Resuscitation => Duration::minutes(0),
                TriageCategory::Emergency => Duration::minutes(15),
                TriageCategory::Urgent => Duration::minutes(60),
                TriageCategory::SemiUrgent => Duration::minutes(120),
                TriageCategory::NonUrgent => Duration::minutes(240),
            };
            
            // Generate recommendations
            let mut recommended_actions = vec![
                RecommendedAction {
                    action: match category {
                        TriageCategory::Resuscitation => "Immediate resuscitation bay".to_string(),
                        TriageCategory::Emergency => "Emergency treatment area".to_string(),
                        TriageCategory::Urgent => "Urgent care area".to_string(),
                        TriageCategory::SemiUrgent => "Treatment area".to_string(),
                        TriageCategory::NonUrgent => "Fast track or waiting area".to_string(),
                    },
                    priority: ActionPriority::Critical,
                    responsible_party: "Triage Nurse".to_string(),
                    time_constraint: Duration::minutes(5),
                },
            ];
            
            // Add specific recommendations from analyses
            recommended_actions.extend(symptom_analysis.recommended_tests.into_iter().map(|test| {
                RecommendedAction {
                    action: format!("Order {}", test),
                    priority: ActionPriority::High,
                    responsible_party: "ED Physician".to_string(),
                    time_constraint: Duration::minutes(30),
                }
            }));
            
            Ok(TriageScore {
                category,
                priority_score: total_score,
                estimated_wait_time: wait_time,
                recommended_actions,
                risk_factors: ai_assessment.risk_factors,
                ai_confidence: ai_assessment.confidence,
                explanation: format!(
                    "Patient triaged as {} based on vital signs (risk: {:.1}), symptoms (severity: {:.1}), medical history (risk: {:.1}), and AI assessment (urgency: {:.1})",
                    match category {
                        TriageCategory::Resuscitation => "Resuscitation",
                        TriageCategory::Emergency => "Emergency",
                        TriageCategory::Urgent => "Urgent",
                        TriageCategory::SemiUrgent => "Semi-Urgent",
                        TriageCategory::NonUrgent => "Non-Urgent",
                    },
                    vital_analysis.risk_score,
                    symptom_analysis.severity_score,
                    history_risk,
                    ai_assessment.urgency_score
                ),
            })
        }
    }
    
    async fn check_and_generate_alerts(&self, patient: &EmergencyPatient, triage_score: &TriageScore) -> Result<(), ApiError> {
        let mut alert_system = self.alert_system.lock().await;
        
        // Check for critical conditions
        if triage_score.category <= TriageCategory::Emergency {
            alert_system.active_alerts.push(Alert {
                alert_id: format!("ALERT_{}", uuid::Uuid::new_v4()),
                alert_type: AlertType::PatientDeterioration,
                patient_id: patient.patient_id.clone(),
                severity: SeverityLevel::Critical,
                message: format!("Critical patient arrival - {} triage", 
                    match triage_score.category {
                        TriageCategory::Resuscitation => "Resuscitation",
                        TriageCategory::Emergency => "Emergency",
                        _ => "Unknown",
                    }
                ),
                timestamp: Utc::now(),
                acknowledged: false,
            });
        }
        
        // Check wait time thresholds
        let resources = self.resource_manager.read().await;
        if let Some(current_wait) = resources.current_wait_times.get(&triage_score.category) {
            if *current_wait > triage_score.estimated_wait_time * 2 {
                alert_system.active_alerts.push(Alert {
                    alert_id: format!("ALERT_{}", uuid::Uuid::new_v4()),
                    alert_type: AlertType::WaitTimeExceeded,
                    patient_id: patient.patient_id.clone(),
                    severity: SeverityLevel::Moderate,
                    message: format!("Wait time exceeding threshold for {} patients", 
                        match triage_score.category {
                            TriageCategory::Resuscitation => "Resuscitation",
                            TriageCategory::Emergency => "Emergency",
                            TriageCategory::Urgent => "Urgent",
                            TriageCategory::SemiUrgent => "Semi-Urgent",
                            TriageCategory::NonUrgent => "Non-Urgent",
                        }
                    ),
                    timestamp: Utc::now(),
                    acknowledged: false,
                });
            }
        }
        
        Ok(())
    }
    
    async fn update_resource_predictions(&self) -> Result<(), ApiError> {
        let queue = self.patient_queue.lock().await;
        let active = self.active_patients.read().await;
        let models = self.prediction_models.read().await;
        
        // Predict resource needs
        let patient_count = queue.len() + active.len();
        let critical_count = queue.iter()
            .filter(|p| p.triage_score.as_ref()
                .map(|ts| ts.category <= TriageCategory::Emergency)
                .unwrap_or(false))
            .count();
        
        // Update wait time predictions
        let mut resources = self.resource_manager.write().await;
        
        // Simple wait time model (would be more sophisticated in production)
        for (category, wait_time) in resources.current_wait_times.iter_mut() {
            let category_patients = queue.iter()
                .filter(|p| p.triage_score.as_ref()
                    .map(|ts| ts.category == *category)
                    .unwrap_or(false))
                .count();
            
            *wait_time = Duration::minutes((category_patients * 30) as i64);
        }
        
        Ok(())
    }
    
    pub async fn get_next_patient(&self) -> Option<EmergencyPatient> {
        let mut queue = self.patient_queue.lock().await;
        queue.pop()
    }
    
    pub async fn update_patient_status(&self, patient_id: &str, status: PatientStatus) -> Result<(), ApiError> {
        let mut active = self.active_patients.write().await;
        
        if let Some(patient) = active.get_mut(patient_id) {
            // Update patient status
            match status {
                PatientStatus::InTreatment { location, physician } => {
                    // Update resource allocation
                    let mut resources = self.resource_manager.write().await;
                    if let Some(beds) = resources.available_beds.get_mut(&location) {
                        *beds = beds.saturating_sub(1);
                    }
                }
                PatientStatus::Discharged { outcome } => {
                    // Remove from active patients
                    active.remove(patient_id);
                    
                    // Free up resources
                    let mut resources = self.resource_manager.write().await;
                    // Update bed availability
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    pub async fn get_department_status(&self) -> DepartmentStatus {
        let queue = self.patient_queue.lock().await;
        let active = self.active_patients.read().await;
        let resources = self.resource_manager.read().await;
        
        let total_beds: u32 = resources.available_beds.values().sum();
        let available_beds: u32 = resources.available_beds.values().sum();
        
        DepartmentStatus {
            total_patients: queue.len() + active.len(),
            patients_by_category: self.count_patients_by_category(&queue, &active),
            average_wait_times: resources.current_wait_times.clone(),
            bed_occupancy: 1.0 - (available_beds as f32 / total_beds as f32),
            staff_utilization: self.calculate_staff_utilization(&resources),
            critical_alerts: self.get_active_critical_alerts().await,
        }
    }
    
    // Helper methods
    fn initialize_bed_availability() -> HashMap<String, u32> {
        [
            ("Resuscitation".to_string(), 4),
            ("Emergency".to_string(), 20),
            ("Urgent".to_string(), 30),
            ("FastTrack".to_string(), 15),
            ("Observation".to_string(), 25),
        ].iter().cloned().collect()
    }
    
    fn initialize_staff() -> HashMap<String, Vec<StaffMember>> {
        let mut staff = HashMap::new();
        
        staff.insert("Physicians".to_string(), vec![
            StaffMember {
                id: "MD001".to_string(),
                role: "Emergency Physician".to_string(),
                specializations: vec!["Trauma".to_string(), "Critical Care".to_string()],
                current_patient: None,
                available: true,
            },
            // Add more staff members
        ]);
        
        staff.insert("Nurses".to_string(), vec![
            StaffMember {
                id: "RN001".to_string(),
                role: "Triage Nurse".to_string(),
                specializations: vec!["Emergency".to_string(), "Pediatrics".to_string()],
                current_patient: None,
                available: true,
            },
            // Add more nurses
        ]);
        
        staff
    }
    
    fn initialize_equipment() -> HashMap<String, EquipmentStatus> {
        [
            ("Ventilators".to_string(), EquipmentStatus {
                equipment_type: "Ventilator".to_string(),
                total_units: 10,
                available_units: 8,
                maintenance_required: vec![],
            }),
            ("Monitors".to_string(), EquipmentStatus {
                equipment_type: "Cardiac Monitor".to_string(),
                total_units: 50,
                available_units: 35,
                maintenance_required: vec![],
            }),
            // Add more equipment
        ].iter().cloned().collect()
    }
    
    fn initialize_wait_times() -> HashMap<TriageCategory, Duration> {
        [
            (TriageCategory::Resuscitation, Duration::minutes(0)),
            (TriageCategory::Emergency, Duration::minutes(15)),
            (TriageCategory::Urgent, Duration::minutes(60)),
            (TriageCategory::SemiUrgent, Duration::minutes(120)),
            (TriageCategory::NonUrgent, Duration::minutes(240)),
        ].iter().cloned().collect()
    }
    
    fn initialize_notification_channels() -> HashMap<String, NotificationChannel> {
        [
            ("SMS".to_string(), NotificationChannel {
                channel_type: "SMS".to_string(),
                recipients: vec![],
                active: true,
            }),
            ("Email".to_string(), NotificationChannel {
                channel_type: "Email".to_string(),
                recipients: vec![],
                active: true,
            }),
            ("Pager".to_string(), NotificationChannel {
                channel_type: "Pager".to_string(),
                recipients: vec![],
                active: true,
            }),
        ].iter().cloned().collect()
    }
    
    fn load_deterioration_model() -> Result<SparseMatrix, ApiError> {
        // Load pre-trained model
        Ok(SparseMatrix::identity(100))
    }
    
    fn load_diagnosis_model() -> Result<SparseMatrix, ApiError> {
        Ok(SparseMatrix::identity(200))
    }
    
    fn load_resource_model() -> Result<SparseMatrix, ApiError> {
        Ok(SparseMatrix::identity(50))
    }
    
    fn load_outcome_model() -> Result<SparseMatrix, ApiError> {
        Ok(SparseMatrix::identity(150))
    }
    
    fn is_red_flag_symptom(&self, symptom: &str) -> bool {
        const RED_FLAGS: &[&str] = &[
            "chest pain",
            "difficulty breathing",
            "severe headache",
            "altered consciousness",
            "severe bleeding",
            "stroke symptoms",
            "severe abdominal pain",
            "anaphylaxis",
        ];
        
        RED_FLAGS.iter().any(|&flag| symptom.to_lowercase().contains(flag))
    }
    
    fn generate_differential_diagnoses(&self, symptoms: &[Symptom], model: &SparseMatrix) -> Result<Vec<String>, ApiError> {
        // Use model to generate differential diagnoses
        // This is a simplified version
        let mut diagnoses = Vec::new();
        
        let symptom_names: Vec<&str> = symptoms.iter().map(|s| s.name.as_str()).collect();
        
        if symptom_names.contains(&"chest pain") {
            diagnoses.push("Acute Coronary Syndrome".to_string());
            diagnoses.push("Pulmonary Embolism".to_string());
            diagnoses.push("Aortic Dissection".to_string());
        }
        
        if symptom_names.contains(&"shortness of breath") {
            diagnoses.push("Pneumonia".to_string());
            diagnoses.push("Congestive Heart Failure".to_string());
            diagnoses.push("Asthma Exacerbation".to_string());
        }
        
        Ok(diagnoses)
    }
    
    fn recommend_diagnostic_tests(&self, diagnoses: &[String]) -> Vec<String> {
        let mut tests = Vec::new();
        
        for diagnosis in diagnoses {
            match diagnosis.as_str() {
                "Acute Coronary Syndrome" => {
                    tests.push("ECG".to_string());
                    tests.push("Troponin".to_string());
                    tests.push("Chest X-ray".to_string());
                }
                "Pulmonary Embolism" => {
                    tests.push("D-dimer".to_string());
                    tests.push("CT Angiography".to_string());
                }
                "Pneumonia" => {
                    tests.push("Chest X-ray".to_string());
                    tests.push("CBC".to_string());
                    tests.push("Blood cultures".to_string());
                }
                _ => {}
            }
        }
        
        tests.sort();
        tests.dedup();
        tests
    }
    
    fn count_patients_by_category(&self, queue: &BinaryHeap<EmergencyPatient>, active: &HashMap<String, EmergencyPatient>) -> HashMap<TriageCategory, usize> {
        let mut counts = HashMap::new();
        
        for patient in queue.iter() {
            if let Some(score) = &patient.triage_score {
                *counts.entry(score.category).or_insert(0) += 1;
            }
        }
        
        for patient in active.values() {
            if let Some(score) = &patient.triage_score {
                *counts.entry(score.category).or_insert(0) += 1;
            }
        }
        
        counts
    }
    
    fn calculate_staff_utilization(&self, resources: &ResourceManager) -> f32 {
        let total_staff: usize = resources.staff_availability.values()
            .map(|staff_list| staff_list.len())
            .sum();
        
        let busy_staff: usize = resources.staff_availability.values()
            .flat_map(|staff_list| staff_list.iter())
            .filter(|staff| !staff.available)
            .count();
        
        if total_staff > 0 {
            busy_staff as f32 / total_staff as f32
        } else {
            0.0
        }
    }
    
    async fn get_active_critical_alerts(&self) -> Vec<Alert> {
        let alert_system = self.alert_system.lock().await;
        alert_system.active_alerts.iter()
            .filter(|alert| alert.severity >= SeverityLevel::Critical && !alert.acknowledged)
            .cloned()
            .collect()
    }
}

// Additional structs used by the system
#[derive(Debug, Clone)]
struct VitalAnalysis {
    critical_findings: Vec<String>,
    risk_score: f32,
    requires_immediate_intervention: bool,
}

#[derive(Debug, Clone)]
struct SymptomAnalysis {
    severity_score: f32,
    critical_symptoms: Vec<String>,
    differential_diagnoses: Vec<String>,
    recommended_tests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatientStatus {
    Waiting,
    InTriage,
    InTreatment { location: String, physician: String },
    UnderObservation,
    AwaitingResults,
    Discharged { outcome: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepartmentStatus {
    pub total_patients: usize,
    pub patients_by_category: HashMap<TriageCategory, usize>,
    pub average_wait_times: HashMap<TriageCategory, Duration>,
    pub bed_occupancy: f32,
    pub staff_utilization: f32,
    pub critical_alerts: Vec<Alert>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_triage_category_ordering() {
        assert!(TriageCategory::Resuscitation < TriageCategory::Emergency);
        assert!(TriageCategory::Emergency < TriageCategory::Urgent);
        assert!(TriageCategory::Urgent < TriageCategory::SemiUrgent);
        assert!(TriageCategory::SemiUrgent < TriageCategory::NonUrgent);
    }
    
    #[test]
    fn test_patient_priority_ordering() {
        let patient1 = EmergencyPatient {
            patient_id: "P1".to_string(),
            arrival_time: Utc::now(),
            chief_complaint: "Chest pain".to_string(),
            vital_signs: VitalSigns {
                heart_rate: 120.0,
                blood_pressure_systolic: 180.0,
                blood_pressure_diastolic: 100.0,
                respiratory_rate: 24.0,
                temperature_celsius: 37.0,
                oxygen_saturation: 95.0,
                pain_scale: 8,
                glasgow_coma_scale: 15,
                blood_glucose: None,
            },
            symptoms: vec![],
            medical_history: MedicalHistory {
                chronic_conditions: vec![],
                medications: vec![],
                allergies: vec![],
                recent_surgeries: vec![],
                immunization_status: HashMap::new(),
            },
            initial_assessment: InitialAssessment {
                performed_by: "RN001".to_string(),
                assessment_time: Utc::now(),
                observations: vec![],
                diagnostic_tests_ordered: vec![],
                immediate_interventions: vec![],
            },
            triage_score: Some(TriageScore {
                category: TriageCategory::Emergency,
                priority_score: 8.5,
                estimated_wait_time: Duration::minutes(15),
                recommended_actions: vec![],
                risk_factors: vec![],
                ai_confidence: 0.95,
                explanation: "Emergency case".to_string(),
            }),
        };
        
        let patient2 = EmergencyPatient {
            patient_id: "P2".to_string(),
            arrival_time: Utc::now() - Duration::minutes(10),
            chief_complaint: "Minor cut".to_string(),
            vital_signs: VitalSigns {
                heart_rate: 75.0,
                blood_pressure_systolic: 120.0,
                blood_pressure_diastolic: 80.0,
                respiratory_rate: 16.0,
                temperature_celsius: 37.0,
                oxygen_saturation: 99.0,
                pain_scale: 2,
                glasgow_coma_scale: 15,
                blood_glucose: None,
            },
            symptoms: vec![],
            medical_history: MedicalHistory {
                chronic_conditions: vec![],
                medications: vec![],
                allergies: vec![],
                recent_surgeries: vec![],
                immunization_status: HashMap::new(),
            },
            initial_assessment: InitialAssessment {
                performed_by: "RN002".to_string(),
                assessment_time: Utc::now(),
                observations: vec![],
                diagnostic_tests_ordered: vec![],
                immediate_interventions: vec![],
            },
            triage_score: Some(TriageScore {
                category: TriageCategory::NonUrgent,
                priority_score: 2.0,
                estimated_wait_time: Duration::minutes(240),
                recommended_actions: vec![],
                risk_factors: vec![],
                ai_confidence: 0.98,
                explanation: "Non-urgent case".to_string(),
            }),
        };
        
        // Emergency patient should have higher priority despite arriving later
        assert!(patient1 > patient2);
    }
    
    #[tokio::test]
    async fn test_vital_signs_analysis() {
        let medical_agent = MedicalAgent::new(Default::default()).await.unwrap();
        let system = EmergencyTriageSystem::new(medical_agent).await.unwrap();
        
        let critical_vitals = VitalSigns {
            heart_rate: 180.0,
            blood_pressure_systolic: 70.0,
            blood_pressure_diastolic: 40.0,
            respiratory_rate: 35.0,
            temperature_celsius: 41.0,
            oxygen_saturation: 85.0,
            pain_scale: 10,
            glasgow_coma_scale: 6,
            blood_glucose: Some(400.0),
        };
        
        let analysis = system.analyze_vital_signs(&critical_vitals).await.unwrap();
        assert!(analysis.requires_immediate_intervention);
        assert!(!analysis.critical_findings.is_empty());
        assert!(analysis.risk_score > 10.0);
    }
    
    #[test]
    fn test_red_flag_symptoms() {
        let medical_agent = MedicalAgent::new(Default::default());
        let system = EmergencyTriageSystem::new(medical_agent.await.unwrap()).await.unwrap();
        
        assert!(system.is_red_flag_symptom("Severe chest pain"));
        assert!(system.is_red_flag_symptom("Difficulty breathing"));
        assert!(system.is_red_flag_symptom("Altered consciousness"));
        assert!(!system.is_red_flag_symptom("Mild headache"));
    }
} 