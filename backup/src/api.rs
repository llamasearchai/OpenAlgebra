use actix_web::{web, App, HttpServer, HttpResponse, Result, middleware::Logger};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::{
    medical::{MedicalDataProcessor, HIPAACompliantStorage, ClinicalValidation},
    dicom::{DICOMProcessor, DICOMSeries},
    models::{MedicalNeuralNetwork, FederatedLearning, ModelType},
    sparse::SparseMatrix,
    utils::{OpenAlgebraConfig, PerformanceMonitor},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DICOMProcessRequest {
    pub file_path: String,
    pub output_format: String,
    pub anonymize: bool,
    pub validate_hipaa: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelTrainRequest {
    pub model_type: String,
    pub dataset_path: String,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub validation_split: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelPredictRequest {
    pub model_path: String,
    pub input_data: Vec<Vec<f64>>,
    pub return_confidence: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FederatedLearningRequest {
    pub model_type: String,
    pub client_count: usize,
    pub rounds: usize,
    pub min_clients: usize,
    pub data_split_strategy: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceBenchmarkRequest {
    pub operation: String,
    pub dataset_size: usize,
    pub iterations: usize,
}

pub struct AppState {
    pub dicom_processor: Arc<Mutex<DICOMProcessor>>,
    pub medical_processor: Arc<Mutex<MedicalDataProcessor>>,
    pub performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    pub config: Arc<OpenAlgebraConfig>,
}

// Health check endpoint
pub async fn health_check() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(ApiResponse::success("OpenAlgebra Medical AI is running")))
}

// DICOM processing endpoints
pub async fn process_dicom(
    data: web::Data<AppState>,
    request: web::Json<DICOMProcessRequest>,
) -> Result<HttpResponse> {
    let mut processor = data.dicom_processor.lock().await;
    
    match processor.load_dicom(&request.file_path) {
        Ok(dicom_data) => {
            let mut result = dicom_data.clone();
            
            if request.anonymize {
                if let Err(e) = processor.anonymize_dicom(&mut result) {
                    return Ok(HttpResponse::BadRequest().json(
                        ApiResponse::<()>::error(format!("Anonymization failed: {}", e))
                    ));
                }
            }
            
            if request.validate_hipaa {
                let validation = processor.validate_hipaa_compliance(&result);
                if !validation.is_compliant {
                    return Ok(HttpResponse::BadRequest().json(
                        ApiResponse::<()>::error("HIPAA compliance validation failed".to_string())
                    ));
                }
            }
            
            let response_data = serde_json::json!({
                "processed": true,
                "patient_id": result.patient_id,
                "study_date": result.study_date,
                "modality": result.modality,
                "image_count": result.images.len(),
            });
            
            Ok(HttpResponse::Ok().json(ApiResponse::success(response_data)))
        },
        Err(e) => Ok(HttpResponse::BadRequest().json(
            ApiResponse::<()>::error(format!("DICOM processing failed: {}", e))
        ))
    }
}

pub async fn get_dicom_metadata(
    data: web::Data<AppState>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let processor = data.dicom_processor.lock().await;
    
    match processor.extract_metadata(&path) {
        Ok(metadata) => Ok(HttpResponse::Ok().json(ApiResponse::success(metadata))),
        Err(e) => Ok(HttpResponse::BadRequest().json(
            ApiResponse::<()>::error(format!("Failed to extract metadata: {}", e))
        ))
    }
}

// Model training and prediction endpoints
pub async fn train_model(
    data: web::Data<AppState>,
    request: web::Json<ModelTrainRequest>,
) -> Result<HttpResponse> {
    let mut medical_processor = data.medical_processor.lock().await;
    let mut monitor = data.performance_monitor.lock().await;
    
    monitor.start_operation("model_training");
    
    let model_type = match request.model_type.as_str() {
        "sparse_cnn" => ModelType::SparseCNN,
        "sparse_transformer" => ModelType::SparseTransformer,
        "graph_neural_network" => ModelType::GraphNeuralNetwork,
        "autoencoder" => ModelType::Autoencoder,
        _ => return Ok(HttpResponse::BadRequest().json(
            ApiResponse::<()>::error("Invalid model type".to_string())
        ))
    };
    
    // Load and preprocess data
    match medical_processor.load_medical_dataset(&request.dataset_path) {
        Ok(dataset) => {
            let mut neural_network = MedicalNeuralNetwork::new(model_type, 1000, 10)?;
            
            // Configure training parameters
            neural_network.configure_training(
                request.learning_rate,
                request.batch_size,
                request.epochs,
            );
            
            // Train the model
            match neural_network.train(&dataset.features, &dataset.labels) {
                Ok(training_metrics) => {
                    monitor.end_operation("model_training");
                    
                    let response_data = serde_json::json!({
                        "model_trained": true,
                        "epochs": request.epochs,
                        "final_loss": training_metrics.final_loss,
                        "accuracy": training_metrics.accuracy,
                        "training_time_ms": monitor.get_operation_time("model_training"),
                    });
                    
                    Ok(HttpResponse::Ok().json(ApiResponse::success(response_data)))
                },
                Err(e) => Ok(HttpResponse::InternalServerError().json(
                    ApiResponse::<()>::error(format!("Training failed: {}", e))
                ))
            }
        },
        Err(e) => Ok(HttpResponse::BadRequest().json(
            ApiResponse::<()>::error(format!("Failed to load dataset: {}", e))
        ))
    }
}

pub async fn predict_model(
    data: web::Data<AppState>,
    request: web::Json<ModelPredictRequest>,
) -> Result<HttpResponse> {
    let mut monitor = data.performance_monitor.lock().await;
    monitor.start_operation("model_prediction");
    
    // Load the trained model (simplified - in practice, you'd load from storage)
    let neural_network = MedicalNeuralNetwork::new(ModelType::SparseCNN, 1000, 10)?;
    
    // Convert input data to SparseMatrix
    let input_matrix = SparseMatrix::from_dense(&request.input_data)?;
    
    match neural_network.predict(&input_matrix) {
        Ok(predictions) => {
            monitor.end_operation("model_prediction");
            
            let response_data = serde_json::json!({
                "predictions": predictions,
                "confidence_scores": if request.return_confidence { 
                    Some(neural_network.get_confidence_scores(&input_matrix).unwrap_or_default())
                } else { 
                    None 
                },
                "prediction_time_ms": monitor.get_operation_time("model_prediction"),
            });
            
            Ok(HttpResponse::Ok().json(ApiResponse::success(response_data)))
        },
        Err(e) => Ok(HttpResponse::InternalServerError().json(
            ApiResponse::<()>::error(format!("Prediction failed: {}", e))
        ))
    }
}

// Federated learning endpoints
pub async fn start_federated_learning(
    data: web::Data<AppState>,
    request: web::Json<FederatedLearningRequest>,
) -> Result<HttpResponse> {
    let mut monitor = data.performance_monitor.lock().await;
    monitor.start_operation("federated_learning");
    
    let model_type = match request.model_type.as_str() {
        "sparse_cnn" => ModelType::SparseCNN,
        "sparse_transformer" => ModelType::SparseTransformer,
        "graph_neural_network" => ModelType::GraphNeuralNetwork,
        _ => return Ok(HttpResponse::BadRequest().json(
            ApiResponse::<()>::error("Invalid model type for federated learning".to_string())
        ))
    };
    
    let mut federated_learning = FederatedLearning::new(
        model_type,
        request.client_count,
        request.min_clients,
    );
    
    // Start federated learning process
    match federated_learning.start_training(request.rounds).await {
        Ok(results) => {
            monitor.end_operation("federated_learning");
            
            let response_data = serde_json::json!({
                "federated_learning_completed": true,
                "rounds": request.rounds,
                "client_count": request.client_count,
                "final_global_accuracy": results.global_accuracy,
                "convergence_round": results.convergence_round,
                "training_time_ms": monitor.get_operation_time("federated_learning"),
            });
            
            Ok(HttpResponse::Ok().json(ApiResponse::success(response_data)))
        },
        Err(e) => Ok(HttpResponse::InternalServerError().json(
            ApiResponse::<()>::error(format!("Federated learning failed: {}", e))
        ))
    }
}

// Performance benchmarking endpoint
pub async fn run_benchmark(
    data: web::Data<AppState>,
    request: web::Json<PerformanceBenchmarkRequest>,
) -> Result<HttpResponse> {
    let mut monitor = data.performance_monitor.lock().await;
    
    let benchmark_results = match request.operation.as_str() {
        "sparse_matrix_multiply" => {
            monitor.start_operation("benchmark_sparse_multiply");
            
            // Create test matrices
            let matrix_a = SparseMatrix::random(request.dataset_size, request.dataset_size, 0.1)?;
            let matrix_b = SparseMatrix::random(request.dataset_size, request.dataset_size, 0.1)?;
            
            let mut total_time = 0u128;
            for _ in 0..request.iterations {
                let start = std::time::Instant::now();
                let _ = matrix_a.multiply(&matrix_b)?;
                total_time += start.elapsed().as_nanos();
            }
            
            monitor.end_operation("benchmark_sparse_multiply");
            
            serde_json::json!({
                "operation": "sparse_matrix_multiply",
                "dataset_size": request.dataset_size,
                "iterations": request.iterations,
                "avg_time_ns": total_time / request.iterations as u128,
                "ops_per_second": (request.iterations as f64) / (total_time as f64 / 1_000_000_000.0),
            })
        },
        "dicom_processing" => {
            monitor.start_operation("benchmark_dicom");
            
            // Benchmark DICOM processing (simplified)
            let processor = data.dicom_processor.lock().await;
            let start = std::time::Instant::now();
            
            // Simulate DICOM processing operations
            for _ in 0..request.iterations {
                // Placeholder for actual DICOM operations
            }
            
            let elapsed = start.elapsed();
            monitor.end_operation("benchmark_dicom");
            
            serde_json::json!({
                "operation": "dicom_processing",
                "iterations": request.iterations,
                "total_time_ms": elapsed.as_millis(),
                "avg_time_ms": elapsed.as_millis() / request.iterations as u128,
            })
        },
        _ => return Ok(HttpResponse::BadRequest().json(
            ApiResponse::<()>::error("Unknown benchmark operation".to_string())
        ))
    };
    
    Ok(HttpResponse::Ok().json(ApiResponse::success(benchmark_results)))
}

// System status endpoint
pub async fn get_system_status(data: web::Data<AppState>) -> Result<HttpResponse> {
    let monitor = data.performance_monitor.lock().await;
    let config = &data.config;
    
    let status = serde_json::json!({
        "system": "OpenAlgebra Medical AI",
        "version": "1.0.0",
        "status": "healthy",
        "gpu_enabled": config.gpu_enabled,
        "hipaa_compliance": config.hipaa_compliance,
        "performance_stats": monitor.get_all_stats(),
        "uptime_ms": monitor.get_uptime().as_millis(),
    });
    
    Ok(HttpResponse::Ok().json(ApiResponse::success(status)))
}

pub async fn start_api_server(config: OpenAlgebraConfig) -> std::io::Result<()> {
    env_logger::init();
    
    let app_state = web::Data::new(AppState {
        dicom_processor: Arc::new(Mutex::new(DICOMProcessor::new()?)),
        medical_processor: Arc::new(Mutex::new(MedicalDataProcessor::new()?)),
        performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
        config: Arc::new(config.clone()),
    });
    
    let bind_address = format!("{}:{}", config.api_host, config.api_port);
    
    log::info!("Starting OpenAlgebra Medical AI API server on {}", bind_address);
    
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(Logger::default())
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .max_age(3600)
            )
            .route("/health", web::get().to(health_check))
            .route("/status", web::get().to(get_system_status))
            .service(
                web::scope("/api/v1")
                    .route("/dicom/process", web::post().to(process_dicom))
                    .route("/dicom/metadata/{path:.*}", web::get().to(get_dicom_metadata))
                    .route("/models/train", web::post().to(train_model))
                    .route("/models/predict", web::post().to(predict_model))
                    .route("/federated/start", web::post().to(start_federated_learning))
                    .route("/benchmark", web::post().to(run_benchmark))
            )
    })
    .bind(&bind_address)?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[tokio::test]
    async fn test_health_check() {
        let app = test::init_service(
            App::new().route("/health", web::get().to(health_check))
        ).await;
        
        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        
        assert!(resp.status().is_success());
    }

    #[tokio::test]
    async fn test_api_response_serialization() {
        let success_response = ApiResponse::success("test data");
        let json = serde_json::to_string(&success_response).unwrap();
        assert!(json.contains("success"));
        assert!(json.contains("test data"));
        
        let error_response: ApiResponse<()> = ApiResponse::error("test error".to_string());
        let json = serde_json::to_string(&error_response).unwrap();
        assert!(json.contains("error"));
        assert!(json.contains("test error"));
    }
} 