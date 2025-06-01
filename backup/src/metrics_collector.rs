//! Advanced metrics collection with Prometheus integration

use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize)]
pub struct SystemMetrics {
    pub timestamp: std::time::SystemTime,
    pub inference_metrics: InferenceMetrics,
    pub resource_metrics: ResourceMetrics,
    pub model_metrics: HashMap<String, ModelMetrics>,
    pub business_metrics: BusinessMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceMetrics {
    pub total_requests: u64,
    pub requests_per_second: f64,
    pub average_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub error_rate: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceMetrics {
    pub cpu_utilization: f64,
    pub memory_usage_gb: f64,
    pub gpu_utilization: f64,
    pub gpu_memory_usage_gb: f64,
    pub network_io_mbps: f64,
    pub disk_io_mbps: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub requests_count: u64,
    pub average_latency: f64,
    pub throughput_tps: f64,
    pub memory_footprint_mb: f64,
    pub accuracy_score: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BusinessMetrics {
    pub revenue_per_request: f64,
    pub cost_per_token: f64,
    pub user_satisfaction_score: f64,
    pub churn_rate: f64,
}

pub struct MetricsCollector {
    system_metrics: Arc<RwLock<SystemMetrics>>,
    latency_histogram: Arc<RwLock<Vec<f64>>>,
    prometheus_metrics: Arc<PrometheusExporter>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            system_metrics: Arc::new(RwLock::new(SystemMetrics::default())),
            latency_histogram: Arc::new(RwLock::new(Vec::new())),
            prometheus_metrics: Arc::new(PrometheusExporter::new()),
        }
    }

    #[tracing::instrument(skip(self))]
    pub async fn record_inference(
        &self,
        model_name: &str,
        latency_ms: f64,
        tokens_generated: usize,
    ) {
        // Update latency histogram
        {
            let mut histogram = self.latency_histogram.write().await;
            histogram.push(latency_ms);
            
            // Keep histogram size manageable
            if histogram.len() > 10000 {
                histogram.drain(0..1000);
            }
        }

        // Update system metrics
        {
            let mut metrics = self.system_metrics.write().await;
            metrics.inference_metrics.total_requests += 1;
            metrics.inference_metrics.tokens_per_second += tokens_generated as f64 / (latency_ms / 1000.0);
            
            // Update model-specific metrics
            let model_metrics = metrics.model_metrics
                .entry(model_name.to_string())
                .or_insert_with(|| ModelMetrics {
                    model_name: model_name.to_string(),
                    requests_count: 0,
                    average_latency: 0.0,
                    throughput_tps: 0.0,
                    memory_footprint_mb: 0.0,
                    accuracy_score: 0.0,
                    cache_hit_rate: 0.0,
                });
            
            model_metrics.requests_count += 1;
            model_metrics.average_latency = 
                (model_metrics.average_latency * (model_metrics.requests_count - 1) as f64 + latency_ms) / 
                model_metrics.requests_count as f64;
        }

        // Export to Prometheus
        self.prometheus_metrics.record_inference(model_name, latency_ms, tokens_generated).await;
    }

    pub async fn calculate_percentiles(&self) -> (f64, f64, f64) {
        let histogram = self.latency_histogram.read().await;
        if histogram.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let mut sorted = histogram.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let len = sorted.len();
        let p50 = sorted[len * 50 / 100];
        let p95 = sorted[len * 95 / 100];
        let p99 = sorted[len * 99 / 100];
        
        (p50, p95, p99)
    }

    pub async fn get_current_metrics(&self) -> SystemMetrics {
        let mut metrics = self.system_metrics.read().await.clone();
        
        // Update percentiles
        let (p50, p95, p99) = self.calculate_percentiles().await;
        metrics.inference_metrics.p50_latency_ms = p50;
        metrics.inference_metrics.p95_latency_ms = p95;
        metrics.inference_metrics.p99_latency_ms = p99;
        
        // Update resource metrics
        metrics.resource_metrics = self.collect_system_resources().await.unwrap_or_default();
        
        metrics
    }

    async fn collect_system_resources(&self) -> Result<ResourceMetrics> {
        Python::with_gil(|py| -> Result<ResourceMetrics> {
            let metrics_code = r#"
import psutil
import pynvml

def collect_system_metrics():
    # CPU and Memory
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_gb = memory.used / (1024**3)
    
    # Network I/O
    net_io = psutil.net_io_counters()
    network_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024**2)
    
    # Disk I/O  
    disk_io = psutil.disk_io_counters()
    disk_mbps = (disk_io.read_bytes + disk_io.write_bytes) / (1024**2)
    
    # GPU metrics
    gpu_util = 0.0
    gpu_memory_gb = 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(util.gpu)
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_gb = mem_info.used / (1024**3)
    except:
        pass
    
    return {
        'cpu_utilization': cpu_percent,
        'memory_usage_gb': memory_gb,
        'gpu_utilization': gpu_util,
        'gpu_memory_usage_gb': gpu_memory_gb,
        'network_io_mbps': network_mbps,
        'disk_io_mbps': disk_mbps
    }

metrics = collect_system_metrics()
            "#;
            
            let result = py.eval(metrics_code, None, None)?;
            let metrics_dict = result.extract::<HashMap<String, f64>>()?;
            
            Ok(ResourceMetrics {
                cpu_utilization: metrics_dict.get("cpu_utilization").copied().unwrap_or(0.0),
                memory_usage_gb: metrics_dict.get("memory_usage_gb").copied().unwrap_or(0.0),
                gpu_utilization: metrics_dict.get("gpu_utilization").copied().unwrap_or(0.0),
                gpu_memory_usage_gb: metrics_dict.get("gpu_memory_usage_gb").copied().unwrap_or(0.0),
                network_io_mbps: metrics_dict.get("network_io_mbps").copied().unwrap_or(0.0),
                disk_io_mbps: metrics_dict.get("disk_io_mbps").copied().unwrap_or(0.0),
            })
        })
    }
}

struct PrometheusExporter {
    // Prometheus client integration would go here
}

impl PrometheusExporter {
    fn new() -> Self {
        Self {}
    }
    
    async fn record_inference(&self, _model_name: &str, _latency_ms: f64, _tokens: usize) {
        // Export metrics to Prometheus
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            timestamp: std::time::SystemTime::now(),
            inference_metrics: InferenceMetrics {
                total_requests: 0,
                requests_per_second: 0.0,
                average_latency_ms: 0.0,
                p50_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                error_rate: 0.0,
                tokens_per_second: 0.0,
            },
            resource_metrics: ResourceMetrics::default(),
            model_metrics: HashMap::new(),
            business_metrics: BusinessMetrics::default(),
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_usage_gb: 0.0,
            gpu_utilization: 0.0,
            gpu_memory_usage_gb: 0.0,
            network_io_mbps: 0.0,
            disk_io_mbps: 0.0,
        }
    }
}

impl Default for BusinessMetrics {
    fn default() -> Self {
        Self {
            revenue_per_request: 0.0,
            cost_per_token: 0.0,
            user_satisfaction_score: 0.0,
            churn_rate: 0.0,
        }
    }
}