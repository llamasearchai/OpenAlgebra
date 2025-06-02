/*!
# Utilities

This module provides utility functions for timing, logging, memory management,
and other common operations in the OpenAlgebra library.
*/

use std::time::{Duration, Instant};
use std::fmt;
use serde::{Deserialize, Serialize};

/// High-precision timer for performance measurements
#[derive(Debug, Clone)]
pub struct Timer {
    start_time: Option<Instant>,
    elapsed_time: Duration,
    name: String,
}

impl Timer {
    /// Create a new timer with a name
    pub fn new(name: &str) -> Self {
        Self {
            start_time: None,
            elapsed_time: Duration::new(0, 0),
            name: name.to_string(),
        }
    }
    
    /// Start the timer
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }
    
    /// Stop the timer and return elapsed time
    pub fn stop(&mut self) -> Duration {
        if let Some(start) = self.start_time.take() {
            let elapsed = start.elapsed();
            self.elapsed_time += elapsed;
            elapsed
        } else {
            Duration::new(0, 0)
        }
    }
    
    /// Get total elapsed time
    pub fn elapsed(&self) -> Duration {
        if let Some(start) = self.start_time {
            self.elapsed_time + start.elapsed()
        } else {
            self.elapsed_time
        }
    }
    
    /// Reset the timer
    pub fn reset(&mut self) {
        self.start_time = None;
        self.elapsed_time = Duration::new(0, 0);
    }
    
    /// Get timer name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Time a closure and return the result and elapsed time
    pub fn time<F, R>(name: &str, f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        (result, elapsed)
    }
}

impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer '{}': {:.3}ms", self.name, self.elapsed().as_secs_f64() * 1000.0)
    }
}

/// Performance profiler for tracking multiple timers
#[derive(Debug, Default)]
pub struct Profiler {
    timers: std::collections::HashMap<String, Timer>,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Start timing an operation
    pub fn start(&mut self, name: &str) {
        let timer = self.timers.entry(name.to_string())
            .or_insert_with(|| Timer::new(name));
        timer.start();
    }
    
    /// Stop timing an operation
    pub fn stop(&mut self, name: &str) -> Option<Duration> {
        self.timers.get_mut(name).map(|timer| timer.stop())
    }
    
    /// Get elapsed time for an operation
    pub fn elapsed(&self, name: &str) -> Option<Duration> {
        self.timers.get(name).map(|timer| timer.elapsed())
    }
    
    /// Reset all timers
    pub fn reset(&mut self) {
        for timer in self.timers.values_mut() {
            timer.reset();
        }
    }
    
    /// Get all timer results
    pub fn results(&self) -> std::collections::HashMap<String, Duration> {
        self.timers.iter()
            .map(|(name, timer)| (name.clone(), timer.elapsed()))
            .collect()
    }
    
    /// Print all timer results
    pub fn print_results(&self) {
        println!("Profiler Results:");
        println!("{:-<50}", "");
        for (name, timer) in &self.timers {
            println!("{:<30} {:>15.3}ms", name, timer.elapsed().as_secs_f64() * 1000.0);
        }
        println!("{:-<50}", "");
    }
}

/// Memory usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub num_allocations: usize,
}

impl MemoryInfo {
    /// Create new memory info
    pub fn new() -> Self {
        Self {
            allocated_bytes: 0,
            peak_bytes: 0,
            num_allocations: 0,
        }
    }
    
    /// Record allocation
    pub fn allocate(&mut self, bytes: usize) {
        self.allocated_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.allocated_bytes);
        self.num_allocations += 1;
    }
    
    /// Record deallocation
    pub fn deallocate(&mut self, bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }
    
    /// Get current memory usage in MB
    pub fn current_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }
    
    /// Get peak memory usage in MB
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Memory: {:.2}MB current, {:.2}MB peak, {} allocations",
               self.current_mb(), self.peak_mb(), self.num_allocations)
    }
}

/// Logger for OpenAlgebra operations
#[derive(Debug, Clone)]
pub struct Logger {
    enabled: bool,
    level: LogLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl Logger {
    /// Create new logger
    pub fn new(level: LogLevel) -> Self {
        Self {
            enabled: true,
            level,
        }
    }
    
    /// Create disabled logger
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            level: LogLevel::Error,
        }
    }
    
    /// Set log level
    pub fn set_level(&mut self, level: LogLevel) {
        self.level = level;
    }
    
    /// Enable/disable logging
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Log error message
    pub fn error(&self, msg: &str) {
        self.log(LogLevel::Error, msg);
    }
    
    /// Log warning message
    pub fn warn(&self, msg: &str) {
        self.log(LogLevel::Warn, msg);
    }
    
    /// Log info message
    pub fn info(&self, msg: &str) {
        self.log(LogLevel::Info, msg);
    }
    
    /// Log debug message
    pub fn debug(&self, msg: &str) {
        self.log(LogLevel::Debug, msg);
    }
    
    /// Log trace message
    pub fn trace(&self, msg: &str) {
        self.log(LogLevel::Trace, msg);
    }
    
    /// Log message at specified level
    fn log(&self, level: LogLevel, msg: &str) {
        if self.enabled && level <= self.level {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            println!("[{}] [{}] {}", timestamp, level, msg);
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            LogLevel::Error => "ERROR",
            LogLevel::Warn => "WARN",
            LogLevel::Info => "INFO",
            LogLevel::Debug => "DEBUG",
            LogLevel::Trace => "TRACE",
        };
        write!(f, "{}", s)
    }
}

/// Error handling utilities
pub fn handle_error<T>(result: crate::Result<T>, logger: &Logger) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(err) => {
            logger.error(&format!("Error: {}", err));
            None
        }
    }
}

/// Format bytes as human readable string
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Format duration as human readable string
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = duration.subsec_millis();
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else if seconds > 0 {
        format!("{}.{:03}s", seconds, millis)
    } else {
        format!("{}ms", millis)
    }
}

/// Progress bar for long-running operations
#[derive(Debug)]
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
    start_time: Instant,
    last_update: Instant,
    update_interval: Duration,
}

impl ProgressBar {
    /// Create new progress bar
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            width: 50,
            start_time: Instant::now(),
            last_update: Instant::now(),
            update_interval: Duration::from_millis(100),
        }
    }
    
    /// Set progress bar width
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }
    
    /// Update progress
    pub fn update(&mut self, current: usize) {
        self.current = current;
        let now = Instant::now();
        
        if now.duration_since(self.last_update) >= self.update_interval || current >= self.total {
            self.render();
            self.last_update = now;
        }
    }
    
    /// Increment progress by 1
    pub fn inc(&mut self) {
        self.update(self.current + 1);
    }
    
    /// Finish progress bar
    pub fn finish(&mut self) {
        self.update(self.total);
        println!();
    }
    
    /// Render progress bar
    fn render(&self) {
        let progress = if self.total > 0 {
            (self.current as f64 / self.total as f64).min(1.0)
        } else {
            0.0
        };
        
        let filled = (progress * self.width as f64) as usize;
        let empty = self.width - filled;
        
        let elapsed = self.start_time.elapsed();
        let eta = if self.current > 0 && progress > 0.0 {
            let rate = self.current as f64 / elapsed.as_secs_f64();
            let remaining = (self.total - self.current) as f64 / rate;
            format_duration(Duration::from_secs(remaining as u64))
        } else {
            "Unknown".to_string()
        };
        
        print!("\r[{}{}] {}/{} ({:.1}%) ETA: {}",
               "=".repeat(filled),
               " ".repeat(empty),
               self.current,
               self.total,
               progress * 100.0,
               eta);
        
        use std::io::{self, Write};
        io::stdout().flush().ok();
    }
}

/// Configuration management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub num_threads: Option<usize>,
    pub memory_limit_mb: Option<usize>,
    pub log_level: String,
    pub enable_profiling: bool,
    pub gpu_enabled: bool,
    pub mpi_enabled: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_threads: None, // Use system default
            memory_limit_mb: None, // No limit
            log_level: "info".to_string(),
            enable_profiling: false,
            gpu_enabled: false,
            mpi_enabled: false,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &str) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> crate::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Get log level enum
    pub fn log_level(&self) -> LogLevel {
        match self.log_level.to_lowercase().as_str() {
            "error" => LogLevel::Error,
            "warn" => LogLevel::Warn,
            "info" => LogLevel::Info,
            "debug" => LogLevel::Debug,
            "trace" => LogLevel::Trace,
            _ => LogLevel::Info,
        }
    }
    
    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }
}

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub git_hash: Option<String>,
    pub build_date: Option<String>,
    pub features: Vec<String>,
}

impl VersionInfo {
    /// Get current version info
    pub fn current() -> Self {
        let mut features = Vec::new();
        
        #[cfg(feature = "gpu-acceleration")]
        features.push("gpu-acceleration".to_string());
        
        #[cfg(feature = "mpi")]
        features.push("mpi".to_string());
        
        #[cfg(feature = "openmp")]
        features.push("openmp".to_string());
        
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_hash: option_env!("GIT_HASH").map(|s| s.to_string()),
            build_date: option_env!("BUILD_DATE").map(|s| s.to_string()),
            features,
        }
    }
}

impl fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenAlgebra v{}", self.version)?;
        
        if let Some(hash) = &self.git_hash {
            write!(f, " ({})", &hash[..8])?;
        }
        
        if let Some(date) = &self.build_date {
            write!(f, " built on {}", date)?;
        }
        
        if !self.features.is_empty() {
            write!(f, " with features: [{}]", self.features.join(", "))?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration as StdDuration;

    #[test]
    fn test_timer() {
        let mut timer = Timer::new("test");
        timer.start();
        thread::sleep(StdDuration::from_millis(10));
        let elapsed = timer.stop();
        
        assert!(elapsed >= StdDuration::from_millis(10));
        assert!(timer.elapsed() >= StdDuration::from_millis(10));
    }
    
    #[test]
    fn test_timer_closure() {
        let (result, elapsed) = Timer::time("test_closure", || {
            thread::sleep(StdDuration::from_millis(5));
            42
        });
        
        assert_eq!(result, 42);
        assert!(elapsed >= StdDuration::from_millis(5));
    }
    
    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();
        
        profiler.start("operation1");
        thread::sleep(StdDuration::from_millis(5));
        profiler.stop("operation1");
        
        profiler.start("operation2");
        thread::sleep(StdDuration::from_millis(10));
        profiler.stop("operation2");
        
        let results = profiler.results();
        assert!(results.contains_key("operation1"));
        assert!(results.contains_key("operation2"));
        assert!(results["operation2"] > results["operation1"]);
    }
    
    #[test]
    fn test_memory_info() {
        let mut mem = MemoryInfo::new();
        mem.allocate(1024 * 1024); // 1MB
        
        assert_eq!(mem.current_mb(), 1.0);
        assert_eq!(mem.peak_mb(), 1.0);
        assert_eq!(mem.num_allocations, 1);
        
        mem.allocate(2 * 1024 * 1024); // 2MB more
        assert_eq!(mem.current_mb(), 3.0);
        assert_eq!(mem.peak_mb(), 3.0);
        
        mem.deallocate(1024 * 1024); // Deallocate 1MB
        assert_eq!(mem.current_mb(), 2.0);
        assert_eq!(mem.peak_mb(), 3.0); // Peak remains
    }
    
    #[test]
    fn test_logger() {
        let logger = Logger::new(LogLevel::Info);
        logger.info("Test message");
        logger.debug("This should not print at Info level");
        
        let disabled_logger = Logger::disabled();
        disabled_logger.error("This should not print");
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(512), "512 B");
    }
    
    #[test]
    fn test_format_duration() {
        let duration = StdDuration::from_millis(1500);
        assert_eq!(format_duration(duration), "1.500s");
        
        let duration = StdDuration::from_secs(75);
        assert_eq!(format_duration(duration), "1m 15s");
        
        let duration = StdDuration::from_secs(3665);
        assert_eq!(format_duration(duration), "1h 1m 5s");
    }
    
    #[test]
    fn test_progress_bar() {
        let mut pb = ProgressBar::new(100);
        pb.update(50);
        pb.finish();
    }
    
    #[test]
    fn test_config() {
        let config = Config::default();
        assert_eq!(config.log_level(), LogLevel::Info);
        assert!(config.num_threads() >= 1);
    }
    
    #[test]
    fn test_version_info() {
        let version = VersionInfo::current();
        assert!(!version.version.is_empty());
    }
} 