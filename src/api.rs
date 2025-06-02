/*!
# API Module

This module provides REST API endpoints for OpenAlgebra operations,
including sparse matrix operations, solver services, and AI agent integration.
*/

use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json as ResponseJson},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use crate::{
    sparse::{COOMatrix, CSRMatrix, SparseMatrix},
    solvers::{ConjugateGradient, GMRES, BiCGSTAB, IterativeSolver},
    tensor::{SparseTensor, DenseTensor, Tensor},
    utils::{Timer, Profiler, MemoryInfo, Logger, LogLevel},
    Result,
};

/// API application state
#[derive(Clone)]
pub struct AppState {
    pub matrices: Arc<RwLock<HashMap<String, MatrixData>>>,
    pub tensors: Arc<RwLock<HashMap<String, TensorData>>>,
    pub profiler: Arc<RwLock<Profiler>>,
    pub memory_info: Arc<RwLock<MemoryInfo>>,
    pub logger: Arc<Logger>,
    pub openai_client: Option<Arc<async_openai::Client<async_openai::config::OpenAIConfig>>>,
}

impl AppState {
    /// Create new application state
    pub fn new() -> Self {
        let logger = Arc::new(Logger::new(LogLevel::Info));
        
        // Initialize OpenAI client if API key is available
        let openai_client = if let Ok(_api_key) = std::env::var("OPENAI_API_KEY") {
            Some(Arc::new(async_openai::Client::new()))
        } else {
            None
        };
        
        Self {
            matrices: Arc::new(RwLock::new(HashMap::new())),
            tensors: Arc::new(RwLock::new(HashMap::new())),
            profiler: Arc::new(RwLock::new(Profiler::new())),
            memory_info: Arc::new(RwLock::new(MemoryInfo::new())),
            logger,
            openai_client,
        }
    }
}

/// Matrix data storage
#[derive(Debug, Clone)]
pub enum MatrixData {
    COO(COOMatrix<f64>),
    CSR(CSRMatrix<f64>),
}

/// Tensor data storage
#[derive(Debug, Clone)]
pub enum TensorData {
    Sparse(SparseTensor<f64>),
    Dense(DenseTensor<f64>),
}

// API Request/Response types
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateMatrixRequest {
    pub name: String,
    pub format: String, // "coo" or "csr"
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<MatrixEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MatrixEntry {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MatrixInfo {
    pub name: String,
    pub format: String,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub density: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolveRequest {
    pub matrix_name: String,
    pub b: Vec<f64>,
    pub solver: String, // "cg", "gmres", "bicgstab"
    pub tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolveResponse {
    pub x: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
    pub solve_time_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateTensorRequest {
    pub name: String,
    pub tensor_type: String, // "sparse" or "dense"
    pub shape: Vec<usize>,
    pub data: Option<Vec<f64>>, // For dense tensors
    pub entries: Option<Vec<TensorEntry>>, // For sparse tensors
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorEntry {
    pub indices: Vec<usize>,
    pub value: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub tensor_type: String,
    pub shape: Vec<usize>,
    pub nnz: usize,
    pub density: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIAgentRequest {
    pub query: String,
    pub context: Option<String>,
    pub matrix_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIAgentResponse {
    pub response: String,
    pub suggestions: Vec<String>,
    pub code_examples: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub features: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

/// Create the API router
pub fn create_router() -> Router<AppState> {
    Router::new()
        // Health endpoints
        .route("/health", get(health_check))
        .route("/version", get(version_info))
        
        // Matrix endpoints
        .route("/matrices", post(create_matrix))
        .route("/matrices", get(list_matrices))
        .route("/matrices/:name", get(get_matrix_info))
        .route("/matrices/:name", delete(delete_matrix))
        .route("/matrices/:name/solve", post(solve_linear_system))
        .route("/matrices/:name/matvec", post(matrix_vector_multiply))
        
        // Tensor endpoints
        .route("/tensors", post(create_tensor))
        .route("/tensors", get(list_tensors))
        .route("/tensors/:name", get(get_tensor_info))
        .route("/tensors/:name", delete(delete_tensor))
        .route("/tensors/:name/operations", post(tensor_operations))
        
        // AI Agent endpoints
        .route("/ai/query", post(ai_query))
        .route("/ai/suggest", post(ai_suggest_optimization))
        .route("/ai/explain", post(ai_explain_algorithm))
        
        // Performance endpoints
        .route("/performance/profile", get(get_performance_profile))
        .route("/performance/memory", get(get_memory_usage))
        
        .layer(CorsLayer::permissive())
}

// Health and info endpoints
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory_info.read().await;
    let features = vec![
        #[cfg(feature = "gpu-acceleration")]
        "gpu-acceleration".to_string(),
        #[cfg(feature = "mpi")]
        "mpi".to_string(),
        #[cfg(feature = "openmp")]
        "openmp".to_string(),
    ];
    
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // TODO: Track actual uptime
        memory_usage_mb: memory.current_mb(),
        features,
    };
    
    ResponseJson(response)
}

async fn version_info() -> impl IntoResponse {
    let version = crate::utils::VersionInfo::current();
    ResponseJson(version)
}

// Matrix endpoints
async fn create_matrix(
    State(state): State<AppState>,
    Json(request): Json<CreateMatrixRequest>,
) -> impl IntoResponse {
    let mut matrices = state.matrices.write().await;
    
    match request.format.as_str() {
        "coo" => {
            let mut matrix = COOMatrix::<f64>::new(request.rows, request.cols);
            for entry in request.entries {
                matrix.insert(entry.row, entry.col, entry.value);
            }
            matrices.insert(request.name.clone(), MatrixData::COO(matrix));
        }
        "csr" => {
            let mut coo = COOMatrix::<f64>::new(request.rows, request.cols);
            for entry in request.entries {
                coo.insert(entry.row, entry.col, entry.value);
            }
            let csr = coo.to_csr();
            matrices.insert(request.name.clone(), MatrixData::CSR(csr));
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                ResponseJson(ErrorResponse {
                    error: "Invalid matrix format".to_string(),
                    code: 400,
                }),
            );
        }
    }
    
    state.logger.info(&format!("Created matrix: {}", request.name));
    
    (StatusCode::CREATED, ResponseJson(serde_json::json!({"status": "created"})))
}

async fn list_matrices(State(state): State<AppState>) -> impl IntoResponse {
    let matrices = state.matrices.read().await;
    let matrix_list: Vec<MatrixInfo> = matrices
        .iter()
        .map(|(name, data)| {
            let (format, shape, nnz) = match data {
                MatrixData::COO(m) => ("coo".to_string(), m.shape(), m.nnz()),
                MatrixData::CSR(m) => ("csr".to_string(), m.shape(), m.nnz()),
            };
            
            let total_elements = (shape.0 * shape.1) as f64;
            let density = if total_elements > 0.0 { nnz as f64 / total_elements } else { 0.0 };
            
            MatrixInfo {
                name: name.clone(),
                format,
                rows: shape.0,
                cols: shape.1,
                nnz,
                density,
            }
        })
        .collect();
    
    ResponseJson(matrix_list)
}

async fn get_matrix_info(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let matrices = state.matrices.read().await;
    
    if let Some(data) = matrices.get(&name) {
        let (format, shape, nnz) = match data {
            MatrixData::COO(m) => ("coo".to_string(), m.shape(), m.nnz()),
            MatrixData::CSR(m) => ("csr".to_string(), m.shape(), m.nnz()),
        };
        
        let total_elements = (shape.0 * shape.1) as f64;
        let density = if total_elements > 0.0 { nnz as f64 / total_elements } else { 0.0 };
        
        let info = MatrixInfo {
            name,
            format,
            rows: shape.0,
            cols: shape.1,
            nnz,
            density,
        };
        
        (StatusCode::OK, ResponseJson(info))
    } else {
        (
            StatusCode::NOT_FOUND,
            ResponseJson(ErrorResponse {
                error: "Matrix not found".to_string(),
                code: 404,
            }),
        )
    }
}

async fn delete_matrix(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut matrices = state.matrices.write().await;
    
    if matrices.remove(&name).is_some() {
        state.logger.info(&format!("Deleted matrix: {}", name));
        (StatusCode::NO_CONTENT, ResponseJson(serde_json::json!({})))
    } else {
        (
            StatusCode::NOT_FOUND,
            ResponseJson(ErrorResponse {
                error: "Matrix not found".to_string(),
                code: 404,
            }),
        )
    }
}

async fn solve_linear_system(
    State(state): State<AppState>,
    Path(matrix_name): Path<String>,
    Json(request): Json<SolveRequest>,
) -> impl IntoResponse {
    let matrices = state.matrices.read().await;
    
    let matrix_data = match matrices.get(&matrix_name) {
        Some(data) => data,
        None => {
            return (
                StatusCode::NOT_FOUND,
                ResponseJson(ErrorResponse {
                    error: "Matrix not found".to_string(),
                    code: 404,
                }),
            );
        }
    };
    
    let mut timer = Timer::new("solve");
    timer.start();
    
    let result = match request.solver.as_str() {
        "cg" => {
            let mut solver = ConjugateGradient::new();
            if let Some(tol) = request.tolerance {
                solver.set_tolerance(tol);
            }
            if let Some(max_iter) = request.max_iterations {
                solver.set_max_iterations(max_iter);
            }
            
            let mut x = vec![0.0; request.b.len()];
            match matrix_data {
                MatrixData::COO(m) => solver.solve(m, &request.b, &mut x),
                MatrixData::CSR(m) => solver.solve(m, &request.b, &mut x),
            }
            .map(|info| (x, info))
        }
        "gmres" => {
            let mut solver = GMRES::new();
            if let Some(tol) = request.tolerance {
                solver.set_tolerance(tol);
            }
            if let Some(max_iter) = request.max_iterations {
                solver.set_max_iterations(max_iter);
            }
            
            let mut x = vec![0.0; request.b.len()];
            match matrix_data {
                MatrixData::COO(m) => solver.solve(m, &request.b, &mut x),
                MatrixData::CSR(m) => solver.solve(m, &request.b, &mut x),
            }
            .map(|info| (x, info))
        }
        "bicgstab" => {
            let mut solver = BiCGSTAB::new();
            if let Some(tol) = request.tolerance {
                solver.set_tolerance(tol);
            }
            if let Some(max_iter) = request.max_iterations {
                solver.set_max_iterations(max_iter);
            }
            
            let mut x = vec![0.0; request.b.len()];
            match matrix_data {
                MatrixData::COO(m) => solver.solve(m, &request.b, &mut x),
                MatrixData::CSR(m) => solver.solve(m, &request.b, &mut x),
            }
            .map(|info| (x, info))
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                ResponseJson(ErrorResponse {
                    error: "Invalid solver type".to_string(),
                    code: 400,
                }),
            );
        }
    };
    
    let solve_time = timer.stop();
    
    match result {
        Ok((x, info)) => {
            let response = SolveResponse {
                x,
                converged: info.converged,
                iterations: info.iterations,
                residual_norm: info.residual_norm,
                solve_time_ms: solve_time.as_secs_f64() * 1000.0,
            };
            (StatusCode::OK, ResponseJson(response))
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ResponseJson(ErrorResponse {
                error: err.to_string(),
                code: 500,
            }),
        ),
    }
}

async fn matrix_vector_multiply(
    State(state): State<AppState>,
    Path(matrix_name): Path<String>,
    Json(x): Json<Vec<f64>>,
) -> impl IntoResponse {
    let matrices = state.matrices.read().await;
    
    let matrix_data = match matrices.get(&matrix_name) {
        Some(data) => data,
        None => {
            return (
                StatusCode::NOT_FOUND,
                ResponseJson(ErrorResponse {
                    error: "Matrix not found".to_string(),
                    code: 404,
                }),
            );
        }
    };
    
    let shape = match matrix_data {
        MatrixData::COO(m) => m.shape(),
        MatrixData::CSR(m) => m.shape(),
    };
    
    let mut y = vec![0.0; shape.0];
    
    let result = match matrix_data {
        MatrixData::COO(m) => m.matvec(&x, &mut y),
        MatrixData::CSR(m) => m.matvec(&x, &mut y),
    };
    
    match result {
        Ok(()) => (StatusCode::OK, ResponseJson(y)),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ResponseJson(ErrorResponse {
                error: err.to_string(),
                code: 500,
            }),
        ),
    }
}

// Tensor endpoints (simplified implementations)
async fn create_tensor(
    State(state): State<AppState>,
    Json(request): Json<CreateTensorRequest>,
) -> impl IntoResponse {
    let mut tensors = state.tensors.write().await;
    
    match request.tensor_type.as_str() {
        "sparse" => {
            let mut tensor = SparseTensor::<f64>::new(request.shape);
            if let Some(entries) = request.entries {
                for entry in entries {
                    if tensor.insert(entry.indices, entry.value).is_err() {
                        return (
                            StatusCode::BAD_REQUEST,
                            ResponseJson(ErrorResponse {
                                error: "Invalid tensor entry".to_string(),
                                code: 400,
                            }),
                        );
                    }
                }
            }
            tensors.insert(request.name.clone(), TensorData::Sparse(tensor));
        }
        "dense" => {
            if let Some(data) = request.data {
                match DenseTensor::from_data(data, request.shape) {
                    Ok(tensor) => {
                        tensors.insert(request.name.clone(), TensorData::Dense(tensor));
                    }
                    Err(err) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            ResponseJson(ErrorResponse {
                                error: err.to_string(),
                                code: 400,
                            }),
                        );
                    }
                }
            } else {
                let tensor = DenseTensor::new(request.shape);
                tensors.insert(request.name.clone(), TensorData::Dense(tensor));
            }
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                ResponseJson(ErrorResponse {
                    error: "Invalid tensor type".to_string(),
                    code: 400,
                }),
            );
        }
    }
    
    (StatusCode::CREATED, ResponseJson(serde_json::json!({"status": "created"})))
}

async fn list_tensors(State(state): State<AppState>) -> impl IntoResponse {
    let tensors = state.tensors.read().await;
    let tensor_list: Vec<TensorInfo> = tensors
        .iter()
        .map(|(name, data)| {
            let (tensor_type, shape, nnz) = match data {
                TensorData::Sparse(t) => ("sparse".to_string(), t.shape().to_vec(), t.nnz()),
                TensorData::Dense(t) => ("dense".to_string(), t.shape().to_vec(), t.nnz()),
            };
            
            let total_elements: usize = shape.iter().product();
            let density = if total_elements > 0 { nnz as f64 / total_elements as f64 } else { 0.0 };
            
            TensorInfo {
                name: name.clone(),
                tensor_type,
                shape,
                nnz,
                density,
            }
        })
        .collect();
    
    ResponseJson(tensor_list)
}

async fn get_tensor_info(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let tensors = state.tensors.read().await;
    
    if let Some(data) = tensors.get(&name) {
        let (tensor_type, shape, nnz) = match data {
            TensorData::Sparse(t) => ("sparse".to_string(), t.shape().to_vec(), t.nnz()),
            TensorData::Dense(t) => ("dense".to_string(), t.shape().to_vec(), t.nnz()),
        };
        
        let total_elements: usize = shape.iter().product();
        let density = if total_elements > 0 { nnz as f64 / total_elements as f64 } else { 0.0 };
        
        let info = TensorInfo {
            name,
            tensor_type,
            shape,
            nnz,
            density,
        };
        
        (StatusCode::OK, ResponseJson(info)).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            ResponseJson(ErrorResponse {
                error: "Tensor not found".to_string(),
                code: 404,
            }),
        ).into_response()
    }
}

async fn delete_tensor(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let mut tensors = state.tensors.write().await;
    
    if tensors.remove(&name).is_some() {
        (StatusCode::NO_CONTENT, ResponseJson(serde_json::json!({})))
    } else {
        (
            StatusCode::NOT_FOUND,
            ResponseJson(ErrorResponse {
                error: "Tensor not found".to_string(),
                code: 404,
            }),
        )
    }
}

async fn tensor_operations(
    State(_state): State<AppState>,
    Path(_name): Path<String>,
    Json(_request): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Placeholder for tensor operations
    (StatusCode::NOT_IMPLEMENTED, ResponseJson(ErrorResponse {
        error: "Tensor operations not implemented yet".to_string(),
        code: 501,
    }))
}

// AI Agent endpoints
async fn ai_query(
    State(state): State<AppState>,
    Json(request): Json<AIAgentRequest>,
) -> impl IntoResponse {
    if let Some(client) = &state.openai_client {
        match process_ai_query(client, &request).await {
            Ok(response) => (StatusCode::OK, ResponseJson(response)),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                ResponseJson(ErrorResponse {
                    error: err.to_string(),
                    code: 500,
                }),
            ),
        }
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            ResponseJson(ErrorResponse {
                error: "OpenAI client not configured".to_string(),
                code: 503,
            }),
        )
    }
}

async fn ai_suggest_optimization(
    State(_state): State<AppState>,
    Json(_request): Json<AIAgentRequest>,
) -> impl IntoResponse {
    // Placeholder for AI optimization suggestions
    let response = AIAgentResponse {
        response: "Consider using GMRES for non-symmetric matrices or CG for symmetric positive definite matrices.".to_string(),
        suggestions: vec![
            "Use preconditioning to accelerate convergence".to_string(),
            "Consider matrix reordering for better cache locality".to_string(),
            "Enable GPU acceleration for large problems".to_string(),
        ],
        code_examples: vec![
            "let solver = GMRES::new();".to_string(),
            "let precond = ILUPreconditioner::ilu0();".to_string(),
        ],
    };
    
    (StatusCode::OK, ResponseJson(response))
}

async fn ai_explain_algorithm(
    State(_state): State<AppState>,
    Json(_request): Json<AIAgentRequest>,
) -> impl IntoResponse {
    // Placeholder for algorithm explanations
    let response = AIAgentResponse {
        response: "The Conjugate Gradient method is an iterative algorithm for solving symmetric positive definite linear systems.".to_string(),
        suggestions: vec![
            "Best for symmetric positive definite matrices".to_string(),
            "Converges in at most n iterations theoretically".to_string(),
            "Works well with preconditioning".to_string(),
        ],
        code_examples: vec![
            "let solver = ConjugateGradient::new();".to_string(),
            "solver.solve(&matrix, &b, &mut x)?;".to_string(),
        ],
    };
    
    (StatusCode::OK, ResponseJson(response))
}

// Performance endpoints
async fn get_performance_profile(State(state): State<AppState>) -> impl IntoResponse {
    let profiler = state.profiler.read().await;
    let results = profiler.results();
    ResponseJson(results)
}

async fn get_memory_usage(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory_info.read().await;
    ResponseJson(memory.clone())
}

// Helper function for AI query processing
async fn process_ai_query(
    _client: &async_openai::Client<async_openai::config::OpenAIConfig>,
    request: &AIAgentRequest,
) -> Result<AIAgentResponse> {
    // Simplified AI processing - in a real implementation, this would use the OpenAI API
    let response = AIAgentResponse {
        response: format!("AI response to: {}", request.query),
        suggestions: vec![
            "Consider using sparse matrix formats for large problems".to_string(),
            "Use iterative solvers for better memory efficiency".to_string(),
        ],
        code_examples: vec![
            "let matrix = COOMatrix::new(1000, 1000);".to_string(),
            "let solver = ConjugateGradient::new();".to_string(),
        ],
    };
    
    Ok(response)
}

/// Start the API server
pub async fn start_server(host: &str, port: u16) -> Result<()> {
    let state = AppState::new();
    let app = create_router().with_state(state);
    
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;
    println!("OpenAlgebra API server listening on http://{}:{}", host, port);
    
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum_test::TestServer;

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = AppState::new();
        let app = create_router().with_state(state);
        let server = TestServer::new(app).unwrap();
        
        let response = server.get("/health").await;
        assert_eq!(response.status_code(), StatusCode::OK);
    }
    
    #[tokio::test]
    async fn test_create_matrix() {
        let state = AppState::new();
        let app = create_router().with_state(state);
        let server = TestServer::new(app).unwrap();
        
        let request = CreateMatrixRequest {
            name: "test_matrix".to_string(),
            format: "coo".to_string(),
            rows: 2,
            cols: 2,
            entries: vec![
                MatrixEntry { row: 0, col: 0, value: 1.0 },
                MatrixEntry { row: 1, col: 1, value: 2.0 },
            ],
        };
        
        let response = server.post("/matrices").json(&request).await;
        assert_eq!(response.status_code(), StatusCode::CREATED);
    }
} 