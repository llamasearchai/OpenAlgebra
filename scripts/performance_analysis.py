#!/usr/bin/env python3
"""
Performance analysis tools for AI inference platform
Comprehensive performance profiling and optimization recommendations
"""

import asyncio
import aiohttp
import requests
import json
import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from datetime import datetime
import psutil
import pandas as pd
from dataclasses import dataclass
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LatencyProfile:
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_dev: float
    
@dataclass
class ThroughputProfile:
    requests_per_second: float
    tokens_per_second: float
    batch_efficiency: float
    concurrency_level: int

class PerformanceAnalyzer:
    """Comprehensive performance analysis for the inference platform"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        self.metrics_history = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def run_comprehensive_analysis(self) -> Dict:
        """Run full performance analysis suite"""
        logger.info("Starting comprehensive performance analysis...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "latency_analysis": await self._analyze_latency_patterns(),
            "throughput_analysis": await self._analyze_throughput_scaling(),
            "resource_utilization": await self._analyze_resource_usage(),
            "bottleneck_detection": await self._detect_bottlenecks(),
            "optimization_recommendations": await self._generate_recommendations(),
        }
        
        await self.generate_performance_report(results)
        return results
    
    def _get_system_info(self) -> Dict:
        """Gather system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": psutil.Process().name(),
            "python_version": psutil.Process().info["python_version"] if "python_version" in psutil.Process().info else "unknown",
        }
    
    async def _analyze_latency_patterns(self) -> Dict:
        """Analyze latency under different conditions"""
        logger.info("Analyzing latency patterns...")
        
        conditions = [
            {"name": "light_load", "concurrent_requests": 1, "payload_size": "small"},
            {"name": "moderate_load", "concurrent_requests": 10, "payload_size": "medium"},
            {"name": "heavy_load", "concurrent_requests": 50, "payload_size": "large"},
            {"name": "burst_load", "concurrent_requests": 100, "payload_size": "mixed"},
        ]
        
        latency_results = []
        
        for condition in conditions:
            logger.info(f"Testing {condition['name']} condition...")
            latencies = await self._measure_latencies(
                concurrent_requests=condition["concurrent_requests"],
                payload_size=condition["payload_size"],
                duration_seconds=60
            )
            
            if latencies:
                profile = LatencyProfile(
                    mean_latency_ms=statistics.mean(latencies),
                    median_latency_ms=statistics.median(latencies),
                    p95_latency_ms=np.percentile(latencies, 95),
                    p99_latency_ms=np.percentile(latencies, 99),
                    std_dev=statistics.stdev(latencies) if len(latencies) > 1 else 0
                )
                
                latency_results.append({
                    "condition": condition["name"],
                    "concurrent_requests": condition["concurrent_requests"],
                    "mean_latency_ms": profile.mean_latency_ms,
                    "median_latency_ms": profile.median_latency_ms,
                    "p95_latency_ms": profile.p95_latency_ms,
                    "p99_latency_ms": profile.p99_latency_ms,
                    "std_dev": profile.std_dev,
                    "jitter": profile.std_dev / profile.mean_latency_ms if profile.mean_latency_ms > 0 else 0,
                })
        
        return {
            "profiles": latency_results,
            "summary": self._summarize_latency_analysis(latency_results)
        }
    
    async def _measure_latencies(self, concurrent_requests: int, payload_size: str, duration_seconds: int) -> List[float]:
        """Measure latencies for given conditions"""
        latencies = []
        end_time = time.time() + duration_seconds
        
        payload = self._generate_payload(payload_size)
        
        async def single_request():
            start_time = time.time()
            try:
                async with self.session.post(
                    f"{self.base_url}/v1/inference",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    await response.json()
                    latency_ms = (time.time() - start_time) * 1000
                    return latency_ms
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None
        
        while time.time() < end_time:
            tasks = [single_request() for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)
            latencies.extend([r for r in results if r is not None])
            await asyncio.sleep(0.1)  # Small delay between batches
        
        return latencies
    
    def _generate_payload(self, size: str) -> Dict:
        """Generate test payload of specified size"""
        sizes = {
            "small": 10,
            "medium": 100,
            "large": 500,
            "mixed": np.random.choice([10, 50, 100, 200, 500])
        }
        
        num_tokens = sizes.get(size, 50)
        
        return {
            "model_name": "benchmark-model",
            "input_tokens": list(range(num_tokens)),
            "max_tokens": min(num_tokens * 2, 100),
            "temperature": 0.7,
        }
    
    async def _analyze_throughput_scaling(self) -> Dict:
        """Analyze throughput scaling with concurrency"""
        logger.info("Analyzing throughput scaling...")
        
        concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128]
        throughput_results = []
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            start_time = time.time()
            total_requests = 0
            total_tokens = 0
            duration = 30  # seconds
            
            async def request_loop():
                nonlocal total_requests, total_tokens
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    try:
                        payload = self._generate_payload("medium")
                        async with self.session.post(
                            f"{self.base_url}/v1/inference",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                total_requests += 1
                                total_tokens += result.get("tokens_generated", 0)
                    except Exception as e:
                        logger.error(f"Request error: {e}")
                    
                    await asyncio.sleep(0.01)
            
            tasks = [request_loop() for _ in range(concurrency)]
            await asyncio.gather(*tasks)
            
            elapsed_time = time.time() - start_time
            
            throughput_results.append({
                "concurrency": concurrency,
                "requests_per_second": total_requests / elapsed_time,
                "tokens_per_second": total_tokens / elapsed_time,
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "duration": elapsed_time,
            })
        
        return {
            "scaling_data": throughput_results,
            "optimal_concurrency": self._find_optimal_concurrency(throughput_results),
            "scalability_coefficient": self._calculate_scalability_coefficient(throughput_results),
        }
    
    async def _analyze_resource_usage(self) -> Dict:
        """Analyze resource utilization patterns"""
        logger.info("Analyzing resource usage...")
        
        scenarios = ["idle", "light_load", "moderate_load", "heavy_load"]
        resource_profiles = []
        
        for scenario in scenarios:
            logger.info(f"Measuring resources for {scenario} scenario...")
            
            # Configure load based on scenario
            if scenario == "idle":
                load_config = {"concurrent": 0, "duration": 10}
            elif scenario == "light_load":
                load_config = {"concurrent": 5, "duration": 30}
            elif scenario == "moderate_load":
                load_config = {"concurrent": 20, "duration": 30}
            else:  # heavy_load
                load_config = {"concurrent": 50, "duration": 30}
            
            # Start load generation if not idle
            if scenario != "idle":
                load_task = asyncio.create_task(
                    self._generate_load(
                        load_config["concurrent"],
                        load_config["duration"]
                    )
                )
            
            # Collect resource metrics
            cpu_samples = []
            memory_samples = []
            gpu_samples = []
            
            for _ in range(load_config["duration"]):
                cpu_samples.append(psutil.cpu_percent(interval=1))
                memory_samples.append(psutil.virtual_memory().percent)
                gpu_samples.append(self._get_gpu_utilization())
            
            if scenario != "idle":
                await load_task
            
            resource_profiles.append({
                "scenario": scenario,
                "avg_cpu_percent": statistics.mean(cpu_samples),
                "max_cpu_percent": max(cpu_samples),
                "avg_memory_percent": statistics.mean(memory_samples),
                "max_memory_percent": max(memory_samples),
                "avg_gpu_percent": statistics.mean(gpu_samples),
                "max_gpu_percent": max(gpu_samples),
                "cpu_variance": statistics.variance(cpu_samples) if len(cpu_samples) > 1 else 0,
                "memory_variance": statistics.variance(memory_samples) if len(memory_samples) > 1 else 0,
            })
        
        return {
            "profiles": resource_profiles,
            "memory_efficiency": self._calculate_memory_efficiency(resource_profiles),
            "cpu_efficiency": self._calculate_cpu_efficiency(resource_profiles),
            "gpu_efficiency": self._calculate_gpu_efficiency(resource_profiles),
        }
    
    async def _detect_bottlenecks(self) -> Dict:
        """Detect performance bottlenecks"""
        logger.info("Detecting bottlenecks...")
        
        bottlenecks = []
        
        # Check latency bottlenecks
        latency_test = await self._measure_latencies(10, "medium", 10)
        if latency_test:
            avg_latency = statistics.mean(latency_test)
            if avg_latency > 500:  # 500ms threshold
                bottlenecks.append({
                    "type": "HIGH_LATENCY",
                    "severity": "HIGH" if avg_latency > 1000 else "MEDIUM",
                    "value": avg_latency,
                    "threshold": 500,
                    "recommendation": "Optimize model inference or scale compute resources"
                })
        
        # Check CPU bottlenecks
        cpu_percent = psutil.cpu_percent(interval=5)
        if cpu_percent > 80:
            bottlenecks.append({
                "type": "CPU_SATURATION",
                "severity": "HIGH" if cpu_percent > 90 else "MEDIUM",
                "value": cpu_percent,
                "threshold": 80,
                "recommendation": "Scale horizontally or optimize CPU-intensive operations"
            })
        
        # Check memory bottlenecks
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            bottlenecks.append({
                "type": "MEMORY_PRESSURE",
                "severity": "HIGH" if memory_percent > 95 else "MEDIUM",
                "value": memory_percent,
                "threshold": 85,
                "recommendation": "Increase memory or optimize memory usage patterns"
            })
        
        # Check I/O bottlenecks
        disk_io = psutil.disk_io_counters()
        if disk_io:
            io_wait = psutil.cpu_times_percent(interval=1).iowait
            if io_wait > 20:
                bottlenecks.append({
                    "type": "IO_BOTTLENECK",
                    "severity": "MEDIUM",
                    "value": io_wait,
                    "threshold": 20,
                    "recommendation": "Optimize disk I/O or use faster storage"
                })
        
        return {
            "bottlenecks": bottlenecks,
            "system_health_score": self._calculate_system_health_score(bottlenecks),
            "priority_issues": self._prioritize_optimizations(bottlenecks),
        }
    
    async def _generate_recommendations(self) -> Dict:
        """Generate optimization recommendations based on analysis"""
        logger.info("Generating optimization recommendations...")
        
        # Get current metrics
        current_metrics = await self._get_current_metrics()
        
        recommendations = []
        
        # Latency recommendations
        if current_metrics["avg_latency_ms"] > 200:
            recommendations.append({
                "category": "LATENCY",
                "priority": "HIGH",
                "recommendation": "Enable model quantization to reduce inference time",
                "expected_improvement": "30-50% latency reduction",
                "implementation_effort": "MEDIUM",
            })
        
        # Throughput recommendations
        if current_metrics["requests_per_second"] < 50:
            recommendations.append({
                "category": "THROUGHPUT",
                "priority": "HIGH",
                "recommendation": "Implement request batching and increase batch size",
                "expected_improvement": "2-3x throughput increase",
                "implementation_effort": "LOW",
            })
        
        # Memory recommendations
        if current_metrics["memory_usage_gb"] > 16:
            recommendations.append({
                "category": "MEMORY",
                "priority": "MEDIUM",
                "recommendation": "Enable gradient checkpointing and optimize tensor allocation",
                "expected_improvement": "40% memory reduction",
                "implementation_effort": "MEDIUM",
            })
        
        # Scaling recommendations
        recommendations.append({
            "category": "SCALING",
            "priority": "MEDIUM",
            "recommendation": "Implement horizontal pod autoscaling with custom metrics",
            "expected_improvement": "Dynamic scaling based on load",
            "implementation_effort": "HIGH",
        })
        
        return {
            "recommendations": recommendations,
            "implementation_roadmap": self._create_implementation_roadmap(recommendations),
            "estimated_total_improvement": self._estimate_total_improvement(recommendations),
        }

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except:
            return 0.0
    
    def _count_outliers(self, data: List[float], z_threshold: float = 3.0) -> int:
        """Count statistical outliers using Z-score method"""
        if len(data) < 3:
            return 0
        
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        
        if std_dev == 0:
            return 0
        
        outliers = 0
        for value in data:
            z_score = abs((value - mean) / std_dev)
            if z_score > z_threshold:
                outliers += 1
        
        return outliers
    
    def _summarize_latency_analysis(self, latency_data: List[Dict]) -> Dict:
        """Create summary statistics for latency analysis"""
        if not latency_data:
            return {}
        
        all_means = [d["mean_latency_ms"] for d in latency_data]
        all_p95s = [d["p95_latency_ms"] for d in latency_data]
        
        return {
            "overall_mean_latency": statistics.mean(all_means),
            "overall_p95_latency": statistics.mean(all_p95s),
            "latency_variability": statistics.stdev(all_means) if len(all_means) > 1 else 0,
            "performance_consistency_score": self._calculate_consistency_score(latency_data),
        }
    
    def _find_optimal_concurrency(self, throughput_data: List[Dict]) -> Dict:
        """Find the optimal concurrency level for maximum efficiency"""
        if not throughput_data:
            return {}
        
        # Find peak throughput
        max_throughput = max(d["tokens_per_second"] for d in throughput_data)
        optimal_point = next(d for d in throughput_data if d["tokens_per_second"] == max_throughput)
        
        # Calculate efficiency (throughput / concurrency)
        for data in throughput_data:
            data["efficiency"] = data["tokens_per_second"] / data["concurrency"]
        
        max_efficiency_point = max(throughput_data, key=lambda x: x["efficiency"])
        
        return {
            "optimal_concurrency": optimal_point["concurrency"],
            "max_throughput_tps": max_throughput,
            "most_efficient_concurrency": max_efficiency_point["concurrency"],
            "efficiency_score": max_efficiency_point["efficiency"],
        }
    
    def _calculate_scalability_coefficient(self, throughput_data: List[Dict]) -> float:
        """Calculate how well the system scales with concurrency"""
        if len(throughput_data) < 2:
            return 0.0
        
        # Linear regression to find scaling trend
        concurrencies = [d["concurrency"] for d in throughput_data]
        throughputs = [d["tokens_per_second"] for d in throughput_data]
        
        # Simple linear correlation coefficient
        n = len(concurrencies)
        sum_x = sum(concurrencies)
        sum_y = sum(throughputs)
        sum_xy = sum(x * y for x, y in zip(concurrencies, throughputs))
        sum_x2 = sum(x * x for x in concurrencies)
        sum_y2 = sum(y * y for y in throughputs)
        
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        if denominator == 0:
            return 0.0
            
        correlation = (n * sum_xy - sum_x * sum_y) / denominator
        
        return max(0.0, correlation)  # Return positive correlation only
    
    def _calculate_memory_efficiency(self, resource_profiles: List[Dict]) -> float:
        """Calculate overall memory efficiency score"""
        if not resource_profiles:
            return 0.0
        
        # Lower memory usage under load = higher efficiency
        idle_memory = next((p["avg_memory_percent"] for p in resource_profiles if p["scenario"] == "idle"), 0)
        heavy_load_memory = next((p["avg_memory_percent"] for p in resource_profiles if p["scenario"] == "heavy_load"), idle_memory)
        
        if idle_memory == 0:
            return 0.0
        
        memory_overhead = (heavy_load_memory - idle_memory) / idle_memory
        efficiency_score = max(0.0, 1.0 - memory_overhead)
        
        return efficiency_score
    
    def _calculate_cpu_efficiency(self, resource_profiles: List[Dict]) -> float:
        """Calculate CPU efficiency score"""
        if not resource_profiles:
            return 0.0
        
        # Find heavy load profile
        heavy_load = next((p for p in resource_profiles if p["scenario"] == "heavy_load"), None)
        if not heavy_load:
            return 0.0
        
        # CPU efficiency = useful work / CPU usage
        # Assuming 100% CPU should handle max theoretical load
        cpu_usage = heavy_load["avg_cpu_percent"]
        if cpu_usage == 0:
            return 0.0
        
        # Normalize to 0-1 scale (assuming 80% CPU is optimal)
        efficiency = min(1.0, 80.0 / cpu_usage)
        
        return efficiency
    
    def _calculate_gpu_efficiency(self, resource_profiles: List[Dict]) -> float:
        """Calculate GPU efficiency score"""
        if not resource_profiles:
            return 0.0
        
        # Find heavy load profile
        heavy_load = next((p for p in resource_profiles if p["scenario"] == "heavy_load"), None)
        if not heavy_load:
            return 0.0
        
        gpu_usage = heavy_load["avg_gpu_percent"]
        if gpu_usage == 0:
            return 0.0  # No GPU or not utilized
        
        # GPU efficiency (should be high under load)
        efficiency = gpu_usage / 100.0
        
        return efficiency
    
    def _detect_memory_leaks(self, memory_profiles: List[Dict]) -> Dict:
        """Detect potential memory leaks by analyzing memory growth patterns"""
        leak_indicators = []
        
        # Check for consistent memory growth across scenarios
        memory_values = [p["avg_memory_percent"] for p in memory_profiles]
        if len(memory_values) >= 3:
            # Check if memory consistently increases
            increases = sum(1 for i in range(1, len(memory_values)) if memory_values[i] > memory_values[i-1])
            if increases >= len(memory_values) - 1:
                leak_indicators.append("Consistent memory growth across load scenarios")
        
        # Check memory stability (high variance indicates potential leaks)
        for profile in memory_profiles:
            if profile.get("memory_variance", 0) > 50:  # High variance threshold
                leak_indicators.append(f"High memory variance in {profile['scenario']} scenario")
        
        return {
            "potential_leak_detected": len(leak_indicators) > 0,
            "leak_indicators": leak_indicators,
            "confidence_score": min(1.0, len(leak_indicators) / 3.0),
        }
    
    def _calculate_system_health_score(self, bottlenecks: List[Dict]) -> float:
        """Calculate overall system health score based on detected bottlenecks"""
        if not bottlenecks:
            return 1.0
        
        severity_weights = {"HIGH": 0.4, "MEDIUM": 0.2, "LOW": 0.1}
        total_penalty = sum(severity_weights.get(b["severity"], 0.1) for b in bottlenecks)
        
        health_score = max(0.0, 1.0 - total_penalty)
        return health_score
    
    def _prioritize_optimizations(self, bottlenecks: List[Dict]) -> List[Dict]:
        """Prioritize optimization efforts based on impact and effort"""
        if not bottlenecks:
            return []
        
        # Sort by severity and potential impact
        priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        sorted_bottlenecks = sorted(
            bottlenecks,
            key=lambda x: priority_order.get(x["severity"], 0),
            reverse=True
        )
        
        return sorted_bottlenecks[:3]  # Top 3 priorities
    
    async def _get_current_metrics(self) -> Dict:
        """Fetch current system metrics"""
        try:
            async with self.session.get(f"{self.base_url}/v1/metrics", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        
        # Fallback metrics
        return {
            "avg_latency_ms": 150,
            "requests_per_second": 25,
            "memory_usage_gb": 8.5,
        }
    
    def _create_implementation_roadmap(self, recommendations: List[Dict]) -> Dict:
        """Create implementation roadmap based on recommendations"""
        roadmap = {
            "immediate": [],
            "short_term": [],
            "long_term": [],
        }
        
        for rec in recommendations:
            effort = rec["implementation_effort"]
            priority = rec["priority"]
            
            if priority == "HIGH" and effort == "LOW":
                roadmap["immediate"].append(rec)
            elif priority == "HIGH" or effort == "MEDIUM":
                roadmap["short_term"].append(rec)
            else:
                roadmap["long_term"].append(rec)
        
        return roadmap
    
    def _estimate_total_improvement(self, recommendations: List[Dict]) -> str:
        """Estimate total improvement from all recommendations"""
        # Simplified estimation - in practice would use more sophisticated modeling
        improvements = {
            "latency": 0,
            "throughput": 0,
            "resource": 0,
        }
        
        for rec in recommendations:
            if rec["category"] == "LATENCY":
                improvements["latency"] += 30  # percentage
            elif rec["category"] == "THROUGHPUT":
                improvements["throughput"] += 100  # percentage
            elif rec["category"] in ["MEMORY", "CPU"]:
                improvements["resource"] += 20  # percentage
        
        return f"Estimated improvements: {improvements['latency']}% latency reduction, {improvements['throughput']}% throughput increase, {improvements['resource']}% resource optimization"
    
    def _calculate_consistency_score(self, latency_data: List[Dict]) -> float:
        """Calculate performance consistency score"""
        if not latency_data:
            return 0.0
        
        std_devs = [d["std_dev"] for d in latency_data]
        means = [d["mean_latency_ms"] for d in latency_data]
        
        # Calculate coefficient of variation for each config
        cvs = [std_dev / mean if mean > 0 else 0 for std_dev, mean in zip(std_devs, means)]
        
        # Lower coefficient of variation = higher consistency
        avg_cv = statistics.mean(cvs) if cvs else 0
        consistency_score = max(0.0, 1.0 - avg_cv)
        
        return consistency_score
    
    async def _generate_load(self, requests_per_second: int, duration: int):
        """Generate synthetic load for testing"""
        interval = 1.0 / requests_per_second if requests_per_second > 0 else 1.0
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Make async request (fire and forget for load generation)
            asyncio.create_task(self._async_inference_request())
            await asyncio.sleep(interval)
    
    async def _async_inference_request(self):
        """Async helper for load generation"""
        try:
            payload = {
                "model_name": "benchmark-gpt",
                "input_tokens": list(range(50)),
                "max_tokens": 25,
                "temperature": 0.7,
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/inference",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                await response.json()
        except:
            pass  # Ignore errors during load generation
    
    async def generate_performance_report(self, results: Dict):
        """Generate comprehensive performance report"""
        report_path = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Inference Platform - Performance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ background: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .high-priority {{ color: #d73527; font-weight: bold; }}
                .medium-priority {{ color: #f57c00; font-weight: bold; }}
                .low-priority {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Analysis Report</h1>
                <p>Generated: {results['timestamp']}</p>
                <p>System Health Score: {results.get('bottleneck_detection', {}).get('system_health_score', 0):.2%}</p>
            </div>
            
            <div class="section">
                <h2>Latency Analysis</h2>
                {self._format_latency_section(results.get('latency_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Throughput Scaling</h2>
                {self._format_throughput_section(results.get('throughput_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Resource Utilization</h2>
                {self._format_resource_section(results.get('resource_utilization', {}))}
            </div>
            
            <div class="section">
                <h2>Optimization Recommendations</h2>
                {self._format_recommendations_section(results.get('optimization_recommendations', {}))}
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        logger.info(f"Performance report generated: {report_path}")
    
    def _format_latency_section(self, latency_data: Dict) -> str:
        """Format latency analysis for HTML report"""
        if not latency_data:
            return "<p>No latency data available</p>"
        
        summary = latency_data.get('summary', {})
        html = f"""
        <div class="metric">
            <strong>Overall Mean Latency:</strong> {summary.get('overall_mean_latency', 0):.2f} ms
        </div>
        <div class="metric">
            <strong>Overall P95 Latency:</strong> {summary.get('overall_p95_latency', 0):.2f} ms
        </div>
        <div class="metric">
            <strong>Performance Consistency Score:</strong> {summary.get('performance_consistency_score', 0):.2%}
        </div>
        """
        return html
    
    def _format_throughput_section(self, throughput_data: Dict) -> str:
        """Format throughput analysis for HTML report"""
        if not throughput_data:
            return "<p>No throughput data available</p>"
        
        optimal = throughput_data.get('optimal_concurrency', {})
        html = f"""
        <div class="metric">
            <strong>Optimal Concurrency:</strong> {optimal.get('optimal_concurrency', 'N/A')}
        </div>
        <div class="metric">
            <strong>Max Throughput:</strong> {optimal.get('max_throughput_tps', 0):.2f} tokens/sec
        </div>
        <div class="metric">
            <strong>Scalability Coefficient:</strong> {throughput_data.get('scalability_coefficient', 0):.3f}
        </div>
        """
        return html
    
    def _format_resource_section(self, resource_data: Dict) -> str:
        """Format resource utilization for HTML report"""
        html = f"""
        <div class="metric">
            <strong>CPU Efficiency:</strong> {resource_data.get('cpu_efficiency', 0):.2%}
        </div>
        <div class="metric">
            <strong>Memory Efficiency:</strong> {resource_data.get('memory_efficiency', 0):.2%}
        </div>
        <div class="metric">
            <strong>GPU Efficiency:</strong> {resource_data.get('gpu_efficiency', 0):.2%}
        </div>
        """
        return html
    
    def _format_recommendations_section(self, recommendations_data: Dict) -> str:
        """Format optimization recommendations for HTML report"""
        if not recommendations_data or 'recommendations' not in recommendations_data:
            return "<p>No recommendations available</p>"
        
        html = "<table><tr><th>Category</th><th>Priority</th><th>Recommendation</th><th>Expected Improvement</th></tr>"
        
        for rec in recommendations_data['recommendations']:
            priority_class = f"{rec['priority'].lower()}-priority"
            html += f"""
            <tr>
                <td>{rec['category']}</td>
                <td class="{priority_class}">{rec['priority']}</td>
                <td>{rec['recommendation']}</td>
                <td>{rec['expected_improvement']}</td>
            </tr>
            """
        
        html += "</table>"
        return html


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='AI Inference Platform Performance Analysis')
    parser.add_argument('--url', default='http://localhost:8080', help='Base URL for the inference platform')
    parser.add_argument('--output', default='performance_analysis.json', help='Output file for results')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--duration', type=int, default=3600, help='Duration for continuous monitoring (seconds)')
    
    args = parser.parse_args()
    
    async with PerformanceAnalyzer(args.url) as analyzer:
        if args.continuous:
            logger.info(f"Starting continuous monitoring for {args.duration} seconds")
            await run_continuous_monitoring(analyzer, args.duration)
        else:
            logger.info("Running comprehensive performance analysis")
            results = await analyzer.run_comprehensive_analysis()
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Analysis complete. Results saved to {args.output}")
            
            # Print summary
            print_analysis_summary(results)


async def run_continuous_monitoring(analyzer: PerformanceAnalyzer, duration: int):
    """Run continuous performance monitoring"""
    end_time = time.time() + duration
    metrics_history = []
    
    while time.time() < end_time:
        try:
            # Collect lightweight metrics every 30 seconds
            start_time = time.time()
            
            # Quick health check
            async with analyzer.session.get(f"{analyzer.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                latency = (time.time() - start_time) * 1000
                
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "health_check_latency_ms": latency,
                    "service_available": response.status == 200,
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "gpu_memory_gb": analyzer._get_gpu_utilization(),
                }
                
                metrics_history.append(metrics)
                
                # Alert on anomalies
                await check_for_anomalies(metrics, analyzer)
                
                await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(30)
    
    # Save monitoring results
    with open(f"continuous_monitoring_{int(time.time())}.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)


async def check_for_anomalies(current_metrics: Dict, analyzer: PerformanceAnalyzer):
    """Check for performance anomalies and trigger alerts"""
    alerts = []
    
    # High latency alert
    if current_metrics["health_check_latency_ms"] > 1000:
        alerts.append({
            "type": "HIGH_LATENCY",
            "severity": "WARNING",
            "message": f"Health check latency: {current_metrics['health_check_latency_ms']:.2f}ms",
        })
    
    # Service unavailable alert
    if not current_metrics["service_available"]:
        alerts.append({
            "type": "SERVICE_DOWN",
            "severity": "CRITICAL",
            "message": "Service health check failed",
        })
    
    # Resource alerts
    if current_metrics["cpu_percent"] > 90:
        alerts.append({
            "type": "HIGH_CPU",
            "severity": "WARNING",
            "message": f"CPU usage: {current_metrics['cpu_percent']:.1f}%",
        })
    
    if current_metrics["memory_percent"] > 85:
        alerts.append({
            "type": "HIGH_MEMORY",
            "severity": "WARNING",
            "message": f"Memory usage: {current_metrics['memory_percent']:.1f}%",
        })
    
    # Send alerts if any
    if alerts:
        await send_alerts(alerts)


async def send_alerts(alerts: List[Dict]):
    """Send alerts to monitoring systems"""
    for alert in alerts:
        logger.warning(f"ALERT [{alert['severity']}] {alert['type']}: {alert['message']}")
        
        # In production, integrate with Slack, PagerDuty, etc.
        # await send_to_slack(alert)
        # await send_to_pagerduty(alert)


def print_analysis_summary(results: Dict):
    """Print a summary of analysis results"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Latency summary
    latency = results.get('latency_analysis', {}).get('summary', {})
    print(f"\nLATENCY METRICS:")
    print(f"   Mean Latency: {latency.get('overall_mean_latency', 0):.2f} ms")
    print(f"   P95 Latency:  {latency.get('overall_p95_latency', 0):.2f} ms")
    print(f"   Consistency:  {latency.get('performance_consistency_score', 0):.1%}")
    
    # Throughput summary
    throughput = results.get('throughput_analysis', {}).get('optimal_concurrency', {})
    print(f"\nTHROUGHPUT METRICS:")
    print(f"   Optimal Concurrency: {throughput.get('optimal_concurrency', 'N/A')}")
    print(f"   Max Throughput:      {throughput.get('max_throughput_tps', 0):.1f} tokens/sec")
    
    # Health summary
    health_score = results.get('bottleneck_detection', {}).get('system_health_score', 0)
    print(f"\nSYSTEM HEALTH:")
    print(f"   Health Score: {health_score:.1%}")
    
    # Top recommendations
    recommendations = results.get('optimization_recommendations', {}).get('recommendations', [])
    if recommendations:
        print(f"\nTOP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. [{rec['priority']}] {rec['recommendation']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())