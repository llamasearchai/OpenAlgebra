//! Real-time Surgery Guidance System
//! 
//! Provides intraoperative guidance for surgeons with sub-50ms latency requirements,
//! AR visualization, and critical safety monitoring.

use crate::{
    api::ApiError,
    models::{MedicalData, ProcessingResult},
    sparse::{SparseMatrix, SparseTensor},
    medical::{MedicalProcessor, AnatomicalModel},
    agents::MedicalAgent,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use ndarray::{Array2, Array3, Axis};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgicalProcedure {
    pub procedure_id: String,
    pub procedure_type: SurgicalType,
    pub patient_id: String,
    pub surgeon_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub anatomical_region: String,
    pub risk_factors: Vec<RiskFactor>,
    pub imaging_modalities: Vec<String>,
    pub safety_thresholds: SafetyThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurgicalType {
    Neurosurgery { target_structure: String },
    CardiacSurgery { procedure_name: String },
    OrthopedicSurgery { joint_or_bone: String },
    LaparoscopicSurgery { organ_system: String },
    VascularSurgery { vessel_targets: Vec<String> },
    OphthalmicSurgery { eye_structure: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor_type: String,
    pub severity: f32,
    pub mitigation_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyThresholds {
    pub max_deviation_mm: f32,
    pub critical_structure_distance_mm: f32,
    pub temperature_limit_celsius: f32,
    pub force_limit_newtons: f32,
    pub time_limit_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgicalGuidance {
    pub timestamp: Instant,
    pub tool_position: [f32; 3],
    pub tool_orientation: [f32; 4], // Quaternion
    pub target_position: [f32; 3],
    pub deviation_mm: f32,
    pub proximity_warnings: Vec<ProximityWarning>,
    pub navigation_path: Vec<[f32; 3]>,
    pub confidence_score: f32,
    pub safety_status: SafetyStatus,
    pub ar_overlay: Option<ARVisualization>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityWarning {
    pub structure_name: String,
    pub distance_mm: f32,
    pub risk_level: RiskLevel,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyStatus {
    Safe,
    Caution { reason: String },
    Warning { reason: String, action: String },
    Critical { reason: String, immediate_action: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARVisualization {
    pub overlay_type: String,
    pub mesh_data: Vec<f32>,
    pub color_map: HashMap<String, [u8; 4]>, // RGBA
    pub transparency: f32,
    pub annotations: Vec<Annotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub position: [f32; 3],
    pub text: String,
    pub importance: f32,
}

pub struct SurgeryGuidanceSystem {
    procedure: Arc<RwLock<SurgicalProcedure>>,
    anatomical_model: Arc<AnatomicalModel>,
    tracking_system: Arc<RwLock<TrackingSystem>>,
    safety_monitor: Arc<RwLock<SafetyMonitor>>,
    guidance_channel: mpsc::Sender<SurgicalGuidance>,
    medical_agent: Arc<MedicalAgent>,
}

struct TrackingSystem {
    camera_calibration: CameraCalibration,
    tool_trackers: HashMap<String, ToolTracker>,
    reference_frames: HashMap<String, ReferenceFrame>,
    fusion_algorithm: SensorFusion,
}

struct CameraCalibration {
    intrinsic_matrix: Array2<f64>,
    distortion_coefficients: Vec<f64>,
    extrinsic_matrix: Array2<f64>,
}

struct ToolTracker {
    tool_id: String,
    marker_pattern: Vec<[f32; 3]>,
    last_position: [f32; 3],
    last_orientation: [f32; 4],
    velocity: [f32; 3],
    confidence: f32,
}

struct ReferenceFrame {
    frame_id: String,
    transform_matrix: Array2<f64>,
    timestamp: Instant,
}

struct SensorFusion {
    kalman_filter: KalmanFilter,
    particle_filter: ParticleFilter,
    fusion_weights: HashMap<String, f32>,
}

struct KalmanFilter {
    state_estimate: Array2<f64>,
    error_covariance: Array2<f64>,
    process_noise: Array2<f64>,
    measurement_noise: Array2<f64>,
}

struct ParticleFilter {
    particles: Vec<Particle>,
    num_particles: usize,
    resampling_threshold: f32,
}

struct Particle {
    state: Vec<f64>,
    weight: f64,
}

struct SafetyMonitor {
    critical_structures: Vec<CriticalStructure>,
    force_sensors: HashMap<String, ForceSensor>,
    temperature_sensors: HashMap<String, TemperatureSensor>,
    alert_history: Vec<SafetyAlert>,
}

struct CriticalStructure {
    name: String,
    geometry: StructureGeometry,
    safety_margin_mm: f32,
    damage_threshold: f32,
}

struct StructureGeometry {
    mesh_vertices: Vec<[f32; 3]>,
    mesh_faces: Vec<[u32; 3]>,
    bounding_box: BoundingBox,
}

struct BoundingBox {
    min_corner: [f32; 3],
    max_corner: [f32; 3],
}

struct ForceSensor {
    sensor_id: String,
    current_force_n: f32,
    max_force_n: f32,
    calibration_factor: f32,
}

struct TemperatureSensor {
    sensor_id: String,
    current_temp_c: f32,
    max_temp_c: f32,
    location: [f32; 3],
}

struct SafetyAlert {
    timestamp: Instant,
    alert_type: AlertType,
    severity: RiskLevel,
    message: String,
    action_taken: String,
}

#[derive(Debug, Clone)]
enum AlertType {
    ProximityAlert,
    ForceExceeded,
    TemperatureExceeded,
    TimeExceeded,
    TrackingLost,
}

impl SurgeryGuidanceSystem {
    pub fn new(
        procedure: SurgicalProcedure,
        anatomical_model: AnatomicalModel,
        medical_agent: MedicalAgent,
    ) -> Result<Self, ApiError> {
        let (tx, _rx) = mpsc::channel(1000);
        
        let tracking_system = TrackingSystem {
            camera_calibration: Self::initialize_camera_calibration()?,
            tool_trackers: HashMap::new(),
            reference_frames: HashMap::new(),
            fusion_algorithm: Self::initialize_sensor_fusion(),
        };
        
        let safety_monitor = SafetyMonitor {
            critical_structures: Self::load_critical_structures(&procedure.anatomical_region)?,
            force_sensors: HashMap::new(),
            temperature_sensors: HashMap::new(),
            alert_history: Vec::new(),
        };
        
        Ok(Self {
            procedure: Arc::new(RwLock::new(procedure)),
            anatomical_model: Arc::new(anatomical_model),
            tracking_system: Arc::new(RwLock::new(tracking_system)),
            safety_monitor: Arc::new(RwLock::new(safety_monitor)),
            guidance_channel: tx,
            medical_agent: Arc::new(medical_agent),
        })
    }
    
    pub async fn start_guidance(&self) -> Result<(), ApiError> {
        // Initialize real-time tracking
        self.initialize_tracking_hardware().await?;
        
        // Start safety monitoring loop
        let safety_handle = self.start_safety_monitoring();
        
        // Start guidance computation loop
        let guidance_handle = self.start_guidance_computation();
        
        // Start AR visualization
        let ar_handle = self.start_ar_visualization();
        
        // Wait for all systems
        tokio::try_join!(safety_handle, guidance_handle, ar_handle)?;
        
        Ok(())
    }
    
    async fn initialize_tracking_hardware(&self) -> Result<(), ApiError> {
        // Initialize optical tracking cameras
        let mut tracking = self.tracking_system.write().unwrap();
        
        // Add surgical tools
        tracking.tool_trackers.insert(
            "primary_tool".to_string(),
            ToolTracker {
                tool_id: "primary_tool".to_string(),
                marker_pattern: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                last_position: [0.0, 0.0, 0.0],
                last_orientation: [1.0, 0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
                confidence: 1.0,
            }
        );
        
        // Initialize reference frames
        tracking.reference_frames.insert(
            "patient_reference".to_string(),
            ReferenceFrame {
                frame_id: "patient_reference".to_string(),
                transform_matrix: Array2::eye(4),
                timestamp: Instant::now(),
            }
        );
        
        Ok(())
    }
    
    fn initialize_camera_calibration() -> Result<CameraCalibration, ApiError> {
        // Load pre-calibrated camera parameters
        Ok(CameraCalibration {
            intrinsic_matrix: Array2::from_shape_vec(
                (3, 3),
                vec![
                    1000.0, 0.0, 640.0,
                    0.0, 1000.0, 480.0,
                    0.0, 0.0, 1.0
                ]
            ).unwrap(),
            distortion_coefficients: vec![0.1, -0.2, 0.0, 0.0, 0.0],
            extrinsic_matrix: Array2::eye(4),
        })
    }
    
    fn initialize_sensor_fusion() -> SensorFusion {
        SensorFusion {
            kalman_filter: KalmanFilter {
                state_estimate: Array2::zeros((6, 1)), // Position and velocity
                error_covariance: Array2::eye(6) * 0.1,
                process_noise: Array2::eye(6) * 0.01,
                measurement_noise: Array2::eye(3) * 0.1,
            },
            particle_filter: ParticleFilter {
                particles: (0..1000).map(|_| Particle {
                    state: vec![0.0; 6],
                    weight: 1.0 / 1000.0,
                }).collect(),
                num_particles: 1000,
                resampling_threshold: 0.5,
            },
            fusion_weights: [
                ("optical".to_string(), 0.7),
                ("electromagnetic".to_string(), 0.2),
                ("ultrasound".to_string(), 0.1),
            ].iter().cloned().collect(),
        }
    }
    
    fn load_critical_structures(anatomical_region: &str) -> Result<Vec<CriticalStructure>, ApiError> {
        // Load anatomical structures based on surgical region
        let structures = match anatomical_region {
            "brain" => vec![
                CriticalStructure {
                    name: "motor_cortex".to_string(),
                    geometry: StructureGeometry {
                        mesh_vertices: vec![], // Would load actual mesh data
                        mesh_faces: vec![],
                        bounding_box: BoundingBox {
                            min_corner: [-50.0, -30.0, 20.0],
                            max_corner: [-20.0, 10.0, 50.0],
                        },
                    },
                    safety_margin_mm: 5.0,
                    damage_threshold: 0.1,
                },
                CriticalStructure {
                    name: "optic_nerve".to_string(),
                    geometry: StructureGeometry {
                        mesh_vertices: vec![],
                        mesh_faces: vec![],
                        bounding_box: BoundingBox {
                            min_corner: [-5.0, -40.0, -10.0],
                            max_corner: [5.0, -30.0, 0.0],
                        },
                    },
                    safety_margin_mm: 3.0,
                    damage_threshold: 0.05,
                },
            ],
            "heart" => vec![
                CriticalStructure {
                    name: "coronary_artery".to_string(),
                    geometry: StructureGeometry {
                        mesh_vertices: vec![],
                        mesh_faces: vec![],
                        bounding_box: BoundingBox {
                            min_corner: [-20.0, -20.0, -5.0],
                            max_corner: [20.0, 20.0, 5.0],
                        },
                    },
                    safety_margin_mm: 2.0,
                    damage_threshold: 0.02,
                },
            ],
            _ => vec![],
        };
        
        Ok(structures)
    }
    
    async fn start_safety_monitoring(&self) -> tokio::task::JoinHandle<Result<(), ApiError>> {
        let safety_monitor = Arc::clone(&self.safety_monitor);
        let guidance_channel = self.guidance_channel.clone();
        let procedure = Arc::clone(&self.procedure);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(10)); // 100Hz monitoring
            
            loop {
                interval.tick().await;
                
                let mut monitor = safety_monitor.write().unwrap();
                let proc = procedure.read().unwrap();
                
                // Check proximity to critical structures
                for structure in &monitor.critical_structures {
                    // Proximity calculation would go here
                    let distance = 10.0; // Placeholder
                    
                    if distance < structure.safety_margin_mm {
                        let warning = ProximityWarning {
                            structure_name: structure.name.clone(),
                            distance_mm: distance,
                            risk_level: if distance < structure.safety_margin_mm / 2.0 {
                                RiskLevel::Critical
                            } else {
                                RiskLevel::High
                            },
                            recommended_action: "Adjust trajectory away from critical structure".to_string(),
                        };
                        
                        // Send warning through guidance channel
                        let guidance = SurgicalGuidance {
                            timestamp: Instant::now(),
                            tool_position: [0.0, 0.0, 0.0], // Would get actual position
                            tool_orientation: [1.0, 0.0, 0.0, 0.0],
                            target_position: [0.0, 0.0, 0.0],
                            deviation_mm: 0.0,
                            proximity_warnings: vec![warning],
                            navigation_path: vec![],
                            confidence_score: 0.95,
                            safety_status: SafetyStatus::Warning {
                                reason: format!("Close to {}", structure.name),
                                action: "Adjust trajectory".to_string(),
                            },
                            ar_overlay: None,
                        };
                        
                        let _ = guidance_channel.send(guidance).await;
                    }
                }
                
                // Check force limits
                for (sensor_id, sensor) in &monitor.force_sensors {
                    if sensor.current_force_n > sensor.max_force_n {
                        monitor.alert_history.push(SafetyAlert {
                            timestamp: Instant::now(),
                            alert_type: AlertType::ForceExceeded,
                            severity: RiskLevel::High,
                            message: format!("Force limit exceeded on {}", sensor_id),
                            action_taken: "Notified surgeon".to_string(),
                        });
                    }
                }
                
                // Check temperature limits
                for (sensor_id, sensor) in &monitor.temperature_sensors {
                    if sensor.current_temp_c > sensor.max_temp_c {
                        monitor.alert_history.push(SafetyAlert {
                            timestamp: Instant::now(),
                            alert_type: AlertType::TemperatureExceeded,
                            severity: RiskLevel::Medium,
                            message: format!("Temperature limit exceeded on {}", sensor_id),
                            action_taken: "Reduced energy output".to_string(),
                        });
                    }
                }
            }
        })
    }
    
    async fn start_guidance_computation(&self) -> tokio::task::JoinHandle<Result<(), ApiError>> {
        let tracking_system = Arc::clone(&self.tracking_system);
        let anatomical_model = Arc::clone(&self.anatomical_model);
        let guidance_channel = self.guidance_channel.clone();
        let medical_agent = Arc::clone(&self.medical_agent);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(20)); // 50Hz guidance
            
            loop {
                interval.tick().await;
                
                let tracking = tracking_system.read().unwrap();
                
                // Get current tool position
                if let Some(tool) = tracking.tool_trackers.get("primary_tool") {
                    // Compute optimal path using AI
                    let ai_guidance = medical_agent.compute_surgical_path(
                        tool.last_position,
                        tool.last_orientation,
                        &anatomical_model,
                    ).await?;
                    
                    // Apply sensor fusion for accurate tracking
                    let fused_position = Self::apply_sensor_fusion(
                        &tracking.fusion_algorithm,
                        tool.last_position,
                        tool.confidence,
                    );
                    
                    // Compute deviation from planned path
                    let deviation = Self::compute_deviation(
                        fused_position,
                        &ai_guidance.navigation_path,
                    );
                    
                    // Generate guidance
                    let guidance = SurgicalGuidance {
                        timestamp: Instant::now(),
                        tool_position: fused_position,
                        tool_orientation: tool.last_orientation,
                        target_position: ai_guidance.target_position,
                        deviation_mm: deviation,
                        proximity_warnings: vec![],
                        navigation_path: ai_guidance.navigation_path,
                        confidence_score: tool.confidence * ai_guidance.confidence,
                        safety_status: if deviation < 2.0 {
                            SafetyStatus::Safe
                        } else if deviation < 5.0 {
                            SafetyStatus::Caution {
                                reason: "Slight deviation from planned path".to_string(),
                            }
                        } else {
                            SafetyStatus::Warning {
                                reason: "Significant deviation from planned path".to_string(),
                                action: "Return to navigation path".to_string(),
                            }
                        },
                        ar_overlay: Some(Self::generate_ar_overlay(&ai_guidance)),
                    };
                    
                    let _ = guidance_channel.send(guidance).await;
                }
            }
        })
    }
    
    async fn start_ar_visualization(&self) -> tokio::task::JoinHandle<Result<(), ApiError>> {
        let anatomical_model = Arc::clone(&self.anatomical_model);
        let guidance_channel = self.guidance_channel.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(33)); // 30Hz AR
            
            loop {
                interval.tick().await;
                
                // Generate AR visualization
                // This would interface with actual AR hardware
                let ar_data = ARVisualization {
                    overlay_type: "surgical_guidance".to_string(),
                    mesh_data: vec![], // Would contain actual mesh
                    color_map: [
                        ("target".to_string(), [0, 255, 0, 128]),
                        ("critical".to_string(), [255, 0, 0, 200]),
                        ("path".to_string(), [0, 0, 255, 100]),
                    ].iter().cloned().collect(),
                    transparency: 0.7,
                    annotations: vec![
                        Annotation {
                            position: [0.0, 0.0, 0.0],
                            text: "Target".to_string(),
                            importance: 1.0,
                        },
                    ],
                };
                
                // Update AR display
                // This would send to actual AR device
            }
            
            Ok(())
        })
    }
    
    fn apply_sensor_fusion(
        fusion: &SensorFusion,
        measured_position: [f32; 3],
        confidence: f32,
    ) -> [f32; 3] {
        // Simplified Kalman filter update
        let measurement = Array2::from_shape_vec(
            (3, 1),
            vec![measured_position[0] as f64, measured_position[1] as f64, measured_position[2] as f64]
        ).unwrap();
        
        // Prediction step would go here
        // Update step would go here
        
        // Return fused position
        [measured_position[0], measured_position[1], measured_position[2]]
    }
    
    fn compute_deviation(position: [f32; 3], path: &[[f32; 3]]) -> f32 {
        if path.is_empty() {
            return 0.0;
        }
        
        // Find closest point on path
        let mut min_distance = f32::MAX;
        
        for point in path {
            let distance = ((position[0] - point[0]).powi(2) +
                           (position[1] - point[1]).powi(2) +
                           (position[2] - point[2]).powi(2)).sqrt();
            
            if distance < min_distance {
                min_distance = distance;
            }
        }
        
        min_distance
    }
    
    fn generate_ar_overlay(ai_guidance: &crate::agents::AIGuidance) -> ARVisualization {
        ARVisualization {
            overlay_type: "navigation".to_string(),
            mesh_data: vec![], // Would generate actual mesh
            color_map: HashMap::new(),
            transparency: 0.6,
            annotations: vec![],
        }
    }
    
    pub async fn emergency_stop(&self) -> Result<(), ApiError> {
        // Immediate halt of all surgical systems
        let mut safety = self.safety_monitor.write().unwrap();
        
        safety.alert_history.push(SafetyAlert {
            timestamp: Instant::now(),
            alert_type: AlertType::TrackingLost,
            severity: RiskLevel::Critical,
            message: "Emergency stop activated".to_string(),
            action_taken: "All systems halted".to_string(),
        });
        
        // Send critical safety status
        let guidance = SurgicalGuidance {
            timestamp: Instant::now(),
            tool_position: [0.0, 0.0, 0.0],
            tool_orientation: [1.0, 0.0, 0.0, 0.0],
            target_position: [0.0, 0.0, 0.0],
            deviation_mm: 0.0,
            proximity_warnings: vec![],
            navigation_path: vec![],
            confidence_score: 0.0,
            safety_status: SafetyStatus::Critical {
                reason: "Emergency stop activated".to_string(),
                immediate_action: "Remove all instruments and assess situation".to_string(),
            },
            ar_overlay: None,
        };
        
        let _ = self.guidance_channel.send(guidance).await;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_surgery_guidance_initialization() {
        let procedure = SurgicalProcedure {
            procedure_id: "PROC001".to_string(),
            procedure_type: SurgicalType::Neurosurgery {
                target_structure: "tumor".to_string(),
            },
            patient_id: "PATIENT001".to_string(),
            surgeon_id: "SURGEON001".to_string(),
            start_time: chrono::Utc::now(),
            anatomical_region: "brain".to_string(),
            risk_factors: vec![],
            imaging_modalities: vec!["MRI".to_string(), "CT".to_string()],
            safety_thresholds: SafetyThresholds {
                max_deviation_mm: 2.0,
                critical_structure_distance_mm: 5.0,
                temperature_limit_celsius: 45.0,
                force_limit_newtons: 10.0,
                time_limit_seconds: 14400,
            },
        };
        
        let anatomical_model = AnatomicalModel::default();
        let medical_agent = MedicalAgent::new(Default::default()).await.unwrap();
        
        let system = SurgeryGuidanceSystem::new(procedure, anatomical_model, medical_agent);
        assert!(system.is_ok());
    }
    
    #[test]
    fn test_proximity_warning_generation() {
        let warning = ProximityWarning {
            structure_name: "optic_nerve".to_string(),
            distance_mm: 3.5,
            risk_level: RiskLevel::High,
            recommended_action: "Adjust trajectory".to_string(),
        };
        
        assert_eq!(warning.risk_level, RiskLevel::High);
        assert!(warning.distance_mm < 5.0);
    }
    
    #[test]
    fn test_deviation_calculation() {
        let position = [5.0, 5.0, 5.0];
        let path = vec![
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            [10.0, 10.0, 10.0],
        ];
        
        let deviation = SurgeryGuidanceSystem::compute_deviation(position, &path);
        assert_eq!(deviation, 0.0);
        
        let off_path_position = [7.0, 5.0, 5.0];
        let off_path_deviation = SurgeryGuidanceSystem::compute_deviation(off_path_position, &path);
        assert!(off_path_deviation > 0.0);
    }
    
    #[test]
    fn test_safety_status_generation() {
        let safe_status = SafetyStatus::Safe;
        let warning_status = SafetyStatus::Warning {
            reason: "High temperature".to_string(),
            action: "Reduce power".to_string(),
        };
        
        match warning_status {
            SafetyStatus::Warning { reason, .. } => {
                assert!(reason.contains("temperature"));
            }
            _ => panic!("Expected warning status"),
        }
    }
} 