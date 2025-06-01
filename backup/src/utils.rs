use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use log::{info, warn, error, debug};

// Configuration management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAlgebraConfig {
    pub gpu_enabled: bool,
    pub num_threads: usize,
    pub memory_limit_gb: f64,
    pub cache_size_mb: usize,
    pub logging_level: String,
    pub output_directory: String,
    pub temp_directory: String,
    pub model_checkpoint_interval: usize,
    pub privacy_mode: bool,
    pub compliance_mode: String, // "HIPAA", "GDPR", "FDA"
    pub performance_monitoring: bool,
    pub distributed_computing: bool,
    pub max_concurrent_jobs: usize,
    // API and agents configuration
    pub api_host: String,
    pub api_port: u16,
    pub enable_api_server: bool,
    pub openai_api_key: Option<String>,
    pub enable_ai_agents: bool,
    pub hipaa_compliance: bool,
}

impl Default for OpenAlgebraConfig {
    fn default() -> Self {
        Self {
            gpu_enabled: true,
            num_threads: num_cpus::get(),
            memory_limit_gb: 8.0,
            cache_size_mb: 512,
            logging_level: "INFO".to_string(),
            output_directory: "./output".to_string(),
            temp_directory: "./temp".to_string(),
            model_checkpoint_interval: 100,
            privacy_mode: true,
            compliance_mode: "HIPAA".to_string(),
            performance_monitoring: true,
            distributed_computing: false,
            max_concurrent_jobs: 4,
            // API and agents configuration
            api_host: "127.0.0.1".to_string(),
            api_port: 8000,
            enable_api_server: true,
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            enable_ai_agents: true,
            hipaa_compliance: true,
        }
    }
}

impl OpenAlgebraConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = serde_json::to_string_pretty(self)?;
        fs::write(path, contents)?;
        Ok(())
    }

    pub fn create_directories(&self) -> Result<()> {
        fs::create_dir_all(&self.output_directory)?;
        fs::create_dir_all(&self.temp_directory)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_threads == 0 {
            return Err(anyhow!("Number of threads must be greater than 0"));
        }
        if self.memory_limit_gb <= 0.0 {
            return Err(anyhow!("Memory limit must be positive"));
        }
        if self.max_concurrent_jobs == 0 {
            return Err(anyhow!("Max concurrent jobs must be greater than 0"));
        }
        Ok(())
    }
}

// Performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub operation_name: String,
    pub start_time: u64,
    pub end_time: u64,
    pub duration_ms: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub gpu_usage_percent: Option<f64>,
    pub throughput: Option<f64>,
    pub success: bool,
    pub error_message: Option<String>,
}

impl PerformanceMetrics {
    pub fn new(operation_name: String) -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            operation_name,
            start_time,
            end_time: 0,
            duration_ms: 0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            gpu_usage_percent: None,
            throughput: None,
            success: false,
            error_message: None,
        }
    }

    pub fn finish(&mut self, success: bool, error_message: Option<String>) {
        self.end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.duration_ms = self.end_time - self.start_time;
        self.success = success;
        self.error_message = error_message;
        self.memory_usage_mb = Self::get_memory_usage();
    }

    fn get_memory_usage() -> f64 {
        // Simplified memory usage calculation
        // In a real implementation, you'd use system APIs
        0.0
    }
}

pub struct PerformanceMonitor {
    metrics: Vec<PerformanceMetrics>,
    config: OpenAlgebraConfig,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            config: OpenAlgebraConfig::default(),
        }
    }

    pub fn with_config(config: OpenAlgebraConfig) -> Self {
        Self {
            metrics: Vec::new(),
            config,
        }
    }

    pub fn start_operation(&mut self, operation_name: String) -> usize {
        let metric = PerformanceMetrics::new(operation_name);
        self.metrics.push(metric);
        self.metrics.len() - 1
    }

    pub fn finish_operation(&mut self, index: usize, success: bool, error_message: Option<String>) {
        if let Some(metric) = self.metrics.get_mut(index) {
            metric.finish(success, error_message);
            
            if self.config.performance_monitoring {
                info!("Operation '{}' completed in {}ms (success: {})", 
                      metric.operation_name, metric.duration_ms, metric.success);
            }
        }
    }

    pub fn get_metrics(&self) -> &[PerformanceMetrics] {
        &self.metrics
    }

    pub fn export_metrics<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.metrics)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn get_average_duration(&self, operation_name: &str) -> Option<f64> {
        let durations: Vec<u64> = self.metrics
            .iter()
            .filter(|m| m.operation_name == operation_name && m.success)
            .map(|m| m.duration_ms)
            .collect();

        if durations.is_empty() {
            None
        } else {
            Some(durations.iter().sum::<u64>() as f64 / durations.len() as f64)
        }
    }

    pub fn start_operation(&mut self, operation_name: &str) {
        let metric = PerformanceMetrics::new(operation_name.to_string());
        self.metrics.push(metric);
    }

    pub fn end_operation(&mut self, operation_name: &str) {
        if let Some(metric) = self.metrics.iter_mut()
            .filter(|m| m.operation_name == operation_name && m.end_time == 0)
            .last() {
            metric.finish(true, None);
        }
    }

    pub fn get_operation_time(&self, operation_name: &str) -> u64 {
        self.metrics.iter()
            .filter(|m| m.operation_name == operation_name && m.success)
            .last()
            .map(|m| m.duration_ms)
            .unwrap_or(0)
    }

    pub fn get_all_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("total_operations".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from(self.metrics.len())));
        
        let successful_ops = self.metrics.iter().filter(|m| m.success).count();
        stats.insert("successful_operations".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from(successful_ops)));
        
        let avg_duration = self.metrics.iter()
            .filter(|m| m.success)
            .map(|m| m.duration_ms)
            .sum::<u64>() as f64 / successful_ops.max(1) as f64;
        stats.insert("average_duration_ms".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from_f64(avg_duration).unwrap_or(serde_json::Number::from(0))));
        
        stats
    }

    pub fn get_uptime(&self) -> std::time::Duration {
        std::time::Duration::from_millis(
            self.metrics.first()
                .map(|m| SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64 - m.start_time)
                .unwrap_or(0)
        )
    }
}

// File I/O utilities
pub struct FileManager {
    config: OpenAlgebraConfig,
}

impl FileManager {
    pub fn new(config: OpenAlgebraConfig) -> Self {
        Self { config }
    }

    pub fn ensure_directory_exists<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            fs::create_dir_all(path)?;
            info!("Created directory: {:?}", path);
        }
        Ok(())
    }

    pub fn safe_write_file<P: AsRef<Path>>(&self, path: P, data: &[u8]) -> Result<()> {
        let path = path.as_ref();
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            self.ensure_directory_exists(parent)?;
        }

        // Write to temporary file first
        let temp_path = path.with_extension("tmp");
        {
            let mut file = BufWriter::new(File::create(&temp_path)?);
            file.write_all(data)?;
            file.flush()?;
        }

        // Atomically rename to final path
        fs::rename(temp_path, path)?;
        info!("Safely wrote file: {:?}", path);
        Ok(())
    }

    pub fn safe_read_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<u8>> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(anyhow!("File does not exist: {:?}", path));
        }

        let mut file = BufReader::new(File::open(path)?);
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        info!("Read file: {:?} ({} bytes)", path, buffer.len());
        Ok(buffer)
    }

    pub fn cleanup_temp_files(&self) -> Result<()> {
        let temp_dir = Path::new(&self.config.temp_directory);
        if temp_dir.exists() {
            for entry in fs::read_dir(temp_dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "tmp" || ext == "temp" {
                            fs::remove_file(&path)?;
                            debug!("Cleaned up temp file: {:?}", path);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn backup_file<P: AsRef<Path>>(&self, original_path: P) -> Result<PathBuf> {
        let original_path = original_path.as_ref();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let backup_path = original_path.with_extension(
            format!("{}.backup.{}", 
                    original_path.extension().unwrap_or_default().to_string_lossy(),
                    timestamp)
        );
        
        fs::copy(original_path, &backup_path)?;
        info!("Created backup: {:?} -> {:?}", original_path, backup_path);
        Ok(backup_path)
    }

    pub fn get_file_size<P: AsRef<Path>>(&self, path: P) -> Result<u64> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.len())
    }

    pub fn list_files_with_extension<P: AsRef<Path>>(&self, directory: P, extension: &str) -> Result<Vec<PathBuf>> {
        let directory = directory.as_ref();
        let mut files = Vec::new();

        if directory.is_dir() {
            for entry in fs::read_dir(directory)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext.to_string_lossy().to_lowercase() == extension.to_lowercase() {
                            files.push(path);
                        }
                    }
                }
            }
        }

        files.sort();
        Ok(files)
    }
}

// Data preprocessing utilities
pub struct DataPreprocessor;

impl DataPreprocessor {
    pub fn normalize_vector(data: &mut [f64], method: &str) -> Result<()> {
        match method {
            "min_max" => Self::min_max_normalize(data),
            "z_score" => Self::z_score_normalize(data),
            "unit_vector" => Self::unit_vector_normalize(data),
            "robust" => Self::robust_normalize(data),
            _ => Err(anyhow!("Unknown normalization method: {}", method)),
        }
    }

    fn min_max_normalize(data: &mut [f64]) -> Result<()> {
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        
        if range == 0.0 {
            return Err(anyhow!("Cannot normalize constant data"));
        }
        
        data.par_iter_mut().for_each(|x| *x = (*x - min_val) / range);
        Ok(())
    }

    fn z_score_normalize(data: &mut [f64]) -> Result<()> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Err(anyhow!("Cannot normalize constant data"));
        }
        
        data.par_iter_mut().for_each(|x| *x = (*x - mean) / std_dev);
        Ok(())
    }

    fn unit_vector_normalize(data: &mut [f64]) -> Result<()> {
        let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm == 0.0 {
            return Err(anyhow!("Cannot normalize zero vector"));
        }
        
        data.par_iter_mut().for_each(|x| *x = *x / norm);
        Ok(())
    }

    fn robust_normalize(data: &mut [f64]) -> Result<()> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = Self::calculate_median(&sorted_data);
        let mad = Self::calculate_mad(&sorted_data, median);
        
        if mad == 0.0 {
            return Err(anyhow!("Cannot normalize with zero MAD"));
        }
        
        data.par_iter_mut().for_each(|x| *x = (*x - median) / mad);
        Ok(())
    }

    fn calculate_median(sorted_data: &[f64]) -> f64 {
        let len = sorted_data.len();
        if len % 2 == 0 {
            (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
        } else {
            sorted_data[len / 2]
        }
    }

    fn calculate_mad(sorted_data: &[f64], median: f64) -> f64 {
        let mut deviations: Vec<f64> = sorted_data.iter().map(|x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self::calculate_median(&deviations)
    }

    pub fn remove_outliers(data: &mut Vec<f64>, method: &str, threshold: f64) -> Result<usize> {
        let original_len = data.len();
        
        match method {
            "iqr" => Self::remove_outliers_iqr(data, threshold),
            "z_score" => Self::remove_outliers_z_score(data, threshold),
            "mad" => Self::remove_outliers_mad(data, threshold),
            _ => return Err(anyhow!("Unknown outlier removal method: {}", method)),
        }
        
        Ok(original_len - data.len())
    }

    fn remove_outliers_iqr(data: &mut Vec<f64>, multiplier: f64) -> Result<()> {
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q1_idx = sorted_data.len() / 4;
        let q3_idx = 3 * sorted_data.len() / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - multiplier * iqr;
        let upper_bound = q3 + multiplier * iqr;
        
        data.retain(|&x| x >= lower_bound && x <= upper_bound);
        Ok(())
    }

    fn remove_outliers_z_score(data: &mut Vec<f64>, threshold: f64) -> Result<()> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Ok(()); // No outliers in constant data
        }
        
        data.retain(|&x| ((x - mean) / std_dev).abs() <= threshold);
        Ok(())
    }

    fn remove_outliers_mad(data: &mut Vec<f64>, threshold: f64) -> Result<()> {
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = Self::calculate_median(&sorted_data);
        let mad = Self::calculate_mad(&sorted_data, median);
        
        if mad == 0.0 {
            return Ok(()); // No outliers with zero MAD
        }
        
        data.retain(|&x| ((x - median) / mad).abs() <= threshold);
        Ok(())
    }

    pub fn impute_missing_values(data: &mut [Option<f64>], method: &str) -> Result<()> {
        match method {
            "mean" => Self::impute_with_mean(data),
            "median" => Self::impute_with_median(data),
            "mode" => Self::impute_with_mode(data),
            "forward_fill" => Self::impute_forward_fill(data),
            "backward_fill" => Self::impute_backward_fill(data),
            "interpolate" => Self::impute_interpolate(data),
            _ => Err(anyhow!("Unknown imputation method: {}", method)),
        }
    }

    fn impute_with_mean(data: &mut [Option<f64>]) -> Result<()> {
        let valid_values: Vec<f64> = data.iter().filter_map(|&x| x).collect();
        if valid_values.is_empty() {
            return Err(anyhow!("No valid values for mean imputation"));
        }
        
        let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
        
        for value in data.iter_mut() {
            if value.is_none() {
                *value = Some(mean);
            }
        }
        Ok(())
    }

    fn impute_with_median(data: &mut [Option<f64>]) -> Result<()> {
        let mut valid_values: Vec<f64> = data.iter().filter_map(|&x| x).collect();
        if valid_values.is_empty() {
            return Err(anyhow!("No valid values for median imputation"));
        }
        
        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = Self::calculate_median(&valid_values);
        
        for value in data.iter_mut() {
            if value.is_none() {
                *value = Some(median);
            }
        }
        Ok(())
    }

    fn impute_with_mode(data: &mut [Option<f64>]) -> Result<()> {
        let valid_values: Vec<f64> = data.iter().filter_map(|&x| x).collect();
        if valid_values.is_empty() {
            return Err(anyhow!("No valid values for mode imputation"));
        }
        
        let mut counts = HashMap::new();
        for &value in &valid_values {
            *counts.entry((value * 1000.0).round() as i64).or_insert(0) += 1;
        }
        
        let mode_key = counts.iter().max_by_key(|(_, &count)| count).unwrap().0;
        let mode = *mode_key as f64 / 1000.0;
        
        for value in data.iter_mut() {
            if value.is_none() {
                *value = Some(mode);
            }
        }
        Ok(())
    }

    fn impute_forward_fill(data: &mut [Option<f64>]) -> Result<()> {
        let mut last_valid = None;
        
        for value in data.iter_mut() {
            if value.is_some() {
                last_valid = *value;
            } else if let Some(fill_value) = last_valid {
                *value = Some(fill_value);
            }
        }
        Ok(())
    }

    fn impute_backward_fill(data: &mut [Option<f64>]) -> Result<()> {
        let mut next_valid = None;
        
        for value in data.iter_mut().rev() {
            if value.is_some() {
                next_valid = *value;
            } else if let Some(fill_value) = next_valid {
                *value = Some(fill_value);
            }
        }
        Ok(())
    }

    fn impute_interpolate(data: &mut [Option<f64>]) -> Result<()> {
        let len = data.len();
        if len < 2 {
            return Ok(());
        }
        
        for i in 0..len {
            if data[i].is_none() {
                // Find the nearest valid values before and after
                let mut before_idx = None;
                let mut after_idx = None;
                
                for j in (0..i).rev() {
                    if data[j].is_some() {
                        before_idx = Some(j);
                        break;
                    }
                }
                
                for j in (i + 1)..len {
                    if data[j].is_some() {
                        after_idx = Some(j);
                        break;
                    }
                }
                
                if let (Some(before), Some(after)) = (before_idx, after_idx) {
                    let before_val = data[before].unwrap();
                    let after_val = data[after].unwrap();
                    let ratio = (i - before) as f64 / (after - before) as f64;
                    let interpolated = before_val + ratio * (after_val - before_val);
                    data[i] = Some(interpolated);
                }
            }
        }
        Ok(())
    }
}

// Validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    pub fn validate_numeric_range(value: f64, min: f64, max: f64, name: &str) -> Result<()> {
        if value < min || value > max {
            return Err(anyhow!("{} must be between {} and {}, got {}", name, min, max, value));
        }
        Ok(())
    }

    pub fn validate_positive(value: f64, name: &str) -> Result<()> {
        if value <= 0.0 {
            return Err(anyhow!("{} must be positive, got {}", name, value));
        }
        Ok(())
    }

    pub fn validate_non_negative(value: f64, name: &str) -> Result<()> {
        if value < 0.0 {
            return Err(anyhow!("{} must be non-negative, got {}", name, value));
        }
        Ok(())
    }

    pub fn validate_dimensions(dims: &[usize], expected_dims: usize, name: &str) -> Result<()> {
        if dims.len() != expected_dims {
            return Err(anyhow!("{} must have {} dimensions, got {}", name, expected_dims, dims.len()));
        }
        for (i, &dim) in dims.iter().enumerate() {
            if dim == 0 {
                return Err(anyhow!("{} dimension {} cannot be zero", name, i));
            }
        }
        Ok(())
    }

    pub fn validate_non_empty<T>(slice: &[T], name: &str) -> Result<()> {
        if slice.is_empty() {
            return Err(anyhow!("{} cannot be empty", name));
        }
        Ok(())
    }

    pub fn validate_same_length<T, U>(slice1: &[T], slice2: &[U], name1: &str, name2: &str) -> Result<()> {
        if slice1.len() != slice2.len() {
            return Err(anyhow!("{} and {} must have the same length ({} != {})", 
                             name1, name2, slice1.len(), slice2.len()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_creation() {
        let config = OpenAlgebraConfig::default();
        assert!(config.gpu_enabled);
        assert!(config.num_threads > 0);
        config.validate().unwrap();
    }

    #[test]
    fn test_file_manager() {
        let config = OpenAlgebraConfig::default();
        let file_manager = FileManager::new(config);
        
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        
        let data = b"Hello, OpenAlgebra!";
        file_manager.safe_write_file(&test_file, data).unwrap();
        
        let read_data = file_manager.safe_read_file(&test_file).unwrap();
        assert_eq!(data, read_data.as_slice());
    }

    #[test]
    fn test_data_preprocessing() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        DataPreprocessor::normalize_vector(&mut data, "min_max").unwrap();
        
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_removal() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier
        let removed = DataPreprocessor::remove_outliers(&mut data, "z_score", 2.0).unwrap();
        
        assert_eq!(removed, 1);
        assert!(!data.contains(&100.0));
    }

    #[test]
    fn test_missing_value_imputation() {
        let mut data = vec![Some(1.0), None, Some(3.0), None, Some(5.0)];
        DataPreprocessor::impute_missing_values(&mut data, "mean").unwrap();
        
        for value in data {
            assert!(value.is_some());
        }
    }

    #[test]
    fn test_validation_utils() {
        ValidationUtils::validate_numeric_range(5.0, 0.0, 10.0, "test").unwrap();
        ValidationUtils::validate_positive(1.0, "test").unwrap();
        ValidationUtils::validate_non_negative(0.0, "test").unwrap();
        ValidationUtils::validate_dimensions(&[10, 20, 30], 3, "test").unwrap();
        ValidationUtils::validate_non_empty(&[1, 2, 3], "test").unwrap();
        ValidationUtils::validate_same_length(&[1, 2], &[3, 4], "test1", "test2").unwrap();
    }

    #[test]
    fn test_performance_monitor() {
        let config = OpenAlgebraConfig::default();
        let mut monitor = PerformanceMonitor::new(config);
        
        let op_index = monitor.start_operation("test_operation".to_string());
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.finish_operation(op_index, true, None);
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.len(), 1);
        assert!(metrics[0].duration_ms >= 10);
        assert!(metrics[0].success);
    }
} 