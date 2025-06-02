/*!
# OpenAlgebra API Server

This binary starts the OpenAlgebra REST API server with all endpoints.
*/

use clap::Parser;
use openalgebra::api;
use std::process;

#[derive(Parser)]
#[command(name = "openalgebra-server")]
#[command(about = "OpenAlgebra API Server - High-Performance Sparse Linear Algebra")]
#[command(version)]
struct Cli {
    /// Host to bind the server to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind the server to
    #[arg(long, default_value = "8000")]
    port: u16,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    // Load configuration if provided
    if let Some(config_path) = &cli.config {
        match openalgebra::utils::Config::from_file(config_path) {
            Ok(config) => {
                tracing::info!("Loaded configuration from {}", config_path);
                tracing::debug!("Config: {:?}", config);
            }
            Err(err) => {
                tracing::error!("Failed to load configuration: {}", err);
                process::exit(1);
            }
        }
    }

    // Initialize OpenAlgebra
    if let Err(err) = openalgebra::init() {
        tracing::error!("Failed to initialize OpenAlgebra: {}", err);
        process::exit(1);
    }

    // Print startup information
    let version = openalgebra::utils::VersionInfo::current();
    tracing::info!("Starting {}", version);
    tracing::info!("Server will bind to http://{}:{}", cli.host, cli.port);

    // Check for OpenAI API key
    if std::env::var("OPENAI_API_KEY").is_ok() {
        tracing::info!("OpenAI integration enabled");
    } else {
        tracing::warn!("OpenAI API key not found - AI features will be disabled");
    }

    // Start the server
    if let Err(err) = api::start_server(&cli.host, cli.port).await {
        tracing::error!("Server error: {}", err);
        process::exit(1);
    }
} 