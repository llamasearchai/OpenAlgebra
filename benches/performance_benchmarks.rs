//! Comprehensive performance benchmarking suite

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ai_inference_platform::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn bench_single_inference(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let (jax_compute, model_registry, inference_engine) = rt.block_on(async {
        let jax_compute = Arc::new(JaxCompute::new().await.unwrap());
        let model_registry = Arc::new(ModelRegistry::new().await.unwrap());
        let inference_engine = Arc::new(
            InferenceEngine::new(jax_compute.clone(), model_registry.clone()).await.unwrap()
        );
        
        // Setup test model
        let test_model = create_benchmark_model();
        model_registry.register_model(test_model).await.unwrap();
        
        (jax_compute, model_registry, inference_engine)
    });

    let mut group = c.benchmark_group("single_inference");
    
    for input_length in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("tokens", input_length),
            input_length,
            |b, &input_len| {
                b.to_async(&rt).iter(|| async {
                    let request = create_benchmark_request(input_len);
                    let response = inference_engine
                        .process_request(black_box(request))
                        .await
                        .unwrap();
                    black_box(response)
                });
            },
        );
    }
    group.finish();
}

fn bench_batch_inference(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let inference_engine = rt.block_on(async {
        let jax_compute = Arc::new(JaxCompute::new().await.unwrap());
        let model_registry = Arc::new(ModelRegistry::new().await.unwrap());
        let inference_engine = Arc::new(
            InferenceEngine::new(jax_compute, model_registry.clone()).await.unwrap()
        );
        
        model_registry.register_model(create_benchmark_model()).await.unwrap();
        inference_engine
    });

    let mut group = c.benchmark_group("batch_inference");
    
    for batch_size in [1, 4, 8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let requests: Vec<_> = (0..batch_size)
                        .map(|_| create_benchmark_request(100))
                        .collect();
                    
                    let responses = inference_engine
                        .batch_inference(black_box(requests))
                        .await
                        .unwrap();
                    black_box(responses)
                });
            },
        );
    }
    group.finish();
}

fn bench_jax_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let jax_compute = rt.block_on(async {
        Arc::new(JaxCompute::new().await.unwrap())
    });

    let mut group = c.benchmark_group("jax_operations");
    
    for matrix_size in [128, 256, 512, 1024, 2048].iter() {
        group.bench_with_input(
            BenchmarkId::new("matrix_multiply", matrix_size),
            matrix_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let input_data: Vec<f32> = (0..size * size)
                        .map(|i| (i as f32) * 0.001)
                        .collect();
                    
                    let request = ComputeRequest {
                        model_id: "benchmark-model".to_string(),
                        input_data: black_box(input_data),
                        batch_size: 1,
                        precision: ComputePrecision::Float32,
                    };
                    
                    let result = jax_compute
                        .compute_inference(black_box(request))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(std::time::Duration::from_secs(30));
    
    // Test memory allocation patterns
    for allocation_size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("allocation", allocation_size),
            allocation_size,
            |b, &size| {
                b.iter(|| {
                    let data: Vec<f32> = black_box((0..size).map(|i| i as f32).collect());
                    let processed: Vec<f32> = data.iter()
                        .map(|&x| x * 2.0 + 1.0)
                        .collect();
                    black_box(processed)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_inference,
    bench_batch_inference,
    bench_jax_operations,
    bench_memory_efficiency
);
criterion_main!(benches);

fn create_benchmark_model() -> crate::model_registry::ModelConfig {
    use crate::model_registry::*;
    
    ModelConfig {
        name: "benchmark-gpt".to_string(),
        version: "1.0.0".to_string(),
        architecture: ModelArchitecture::Transformer {
            layers: 24,
            heads: 16,
            dim: 1024,
            vocab_size: 50257,
        },
        parameters: ModelParameters {
            total_params: 1_300_000_000,
            active_params: 1_300_000_000,
            memory_footprint_mb: 2600,
            flops_per_token: 10_400_000,
        },
        precision: crate::jax_bridge::ComputePrecision::Float32,
        deployment_config: DeploymentConfig {
            min_replicas: 2,
            max_replicas: 16,
            auto_scaling: true,
            gpu_memory_fraction: 0.9,
            tensor_parallelism: 4,
            pipeline_parallelism: 2,
        },
        performance_profile: PerformanceProfile {
            throughput_tokens_per_sec: 2500.0,
            latency_p50_ms: 25.0,
            latency_p99_ms: 100.0,
            memory_efficiency: 0.92,
            accuracy_benchmarks: std::collections::HashMap::from([
                ("hellaswag".to_string(), 0.89),
                ("mmlu".to_string(), 0.82),
                ("humaneval".to_string(), 0.71),
                ("gsm8k".to_string(), 0.68),
            ]),
        },
    }
}

fn create_benchmark_request(input_length: usize) -> crate::inference_engine::InferenceRequest {
    use crate::inference_engine::InferenceRequest;
    use uuid::Uuid;
    
    InferenceRequest {
        id: Uuid::new_v4(),
        model_name: "benchmark-gpt".to_string(),
        input_tokens: (0..input_length).map(|i| (i % 50000) as u32).collect(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        batch_id: None,
    }
}