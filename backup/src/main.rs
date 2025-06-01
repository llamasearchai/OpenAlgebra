//! High-Performance AI Inference Platform
//! Rust + JAX integration for production ML workloads

use anyhow::Result;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, instrument};

mod inference_engine;
mod jax_bridge;
mod model_registry;
mod request_handler;
mod metrics_collector;
mod distributed_cache;

use crate::{
    inference_engine::InferenceEngine,
    jax_bridge::JaxCompute,
    model_registry::ModelRegistry,
    request_handler::RequestHandler,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting AI Inference Platform");

    // Initialize core components
    let jax_compute = Arc::new(JaxCompute::new().await?);
    let model_registry = Arc::new(ModelRegistry::new().await?);
    let inference_engine = Arc::new(
        InferenceEngine::new(jax_compute.clone(), model_registry.clone()).await?
    );
    
    // Start HTTP server
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    let handler = RequestHandler::new(inference_engine);
    
    info!("Server listening on 0.0.0.0:8080");
    
    loop {
        let (stream, _) = listener.accept().await?;
        let handler = handler.clone();
        
        tokio::spawn(async move {
            if let Err(e) = handler.handle_connection(stream).await {
                tracing::error!("Connection error: {}", e);
            }
        });
    }
}
