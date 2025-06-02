#!/usr/bin/env python3
"""
Production-grade load testing suite for AI inference platform
Comprehensive testing with realistic workload patterns
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import Dict, List, Tuple
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    base_url: str
    duration_seconds: int
    ramp_up_seconds: int
    max_concurrent_users: int
    request_patterns: List[Dict]
    
@dataclass
class TestResult:
    timestamp: float
    latency_ms: float
    status_code: int
    error_message: str = None
    tokens_generated: int = 0
    user_id: str = ""

class LoadTestRunner:
    """Advanced load testing with realistic user behavior patterns"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = []
        self.active_users = 0
        self.test_start_time = None
        
    async def run_load_test(self) -> Dict:
        """Execute comprehensive load test"""
        logger.info(f"Starting load test: {self.config.max_concurrent_users} users, {self.config.duration_seconds}s duration")
        
        self.test_start_time = time.time()
        results = {
            "config": self.config.__dict__,
            "start_time": self.test_start_time,
            "results": [],
            "summary": {},
        }
        
        # Run different test scenarios
        scenarios = [
            self._run_baseline_test(),
            self._run_spike_test(),
            self._run_stress_test(),
            self._run_endurance_test(),
        ]
        
        for scenario in scenarios:
            scenario_results = await scenario
            results["results"].append(scenario_results)
        
        results["summary"] = self._analyze_overall_results(results["results"])
        await self._generate_load_test_report(results)
        
        return results
    
    async def _run_baseline_test(self) -> Dict:
        """Baseline performance test with normal load"""
        logger.info("Running baseline performance test...")
        
        concurrent_users = min(10, self.config.max_concurrent_users // 4)
        duration = 300  # 5 minutes
        
        return await self._execute_scenario(
            "baseline",
            concurrent_users,
            duration,
            self._normal_user_pattern
        )
    
    async def _run_spike_test(self) -> Dict:
        """Spike test - sudden load increase"""
        logger.info("Running spike test...")
        
        # Start with low load, then spike
        scenarios = [
            (5, 60, self._normal_user_pattern),  # 1 min warm-up
            (self.config.max_concurrent_users, 120, self._aggressive_user_pattern),  # 2 min spike
            (10, 60, self._normal_user_pattern),  # 1 min cool-down
        ]
        
        spike_results = []
        for users, duration, pattern in scenarios:
            result = await self._execute_scenario(f"spike_{users}users", users, duration, pattern)
            spike_results.append(result)
        
        return {
            "scenario": "spike_test",
            "phases": spike_results,
            "analysis": self._analyze_spike_behavior(spike_results),
        }
    
    async def _run_stress_test(self) -> Dict:
        """Stress test - gradually increase load beyond normal capacity"""
        logger.info("Running stress test...")
        
        stress_results = []
        max_users = self.config.max_concurrent_users
        
        # Gradually increase load
        for users in range(5, max_users + 1, max(1, max_users // 10)):
            logger.info(f"Stress testing with {users} concurrent users")
            
            result = await self._execute_scenario(
                f"stress_{users}users",
                users,
                120,  # 2 minutes per level
                self._stress_user_pattern
            )
            stress_results.append(result)
            
            # Break if error rate > 10%
            if result["metrics"]["error_rate"] > 0.1:
                logger.warning(f"High error rate detected at {users} users, stopping stress test")
                break
        
        return {
            "scenario": "stress_test",
            "levels": stress_results,
            "breaking_point": self._find_breaking_point(stress_results),
        }
    
    async def _run_endurance_test(self) -> Dict:
        """Endurance test - sustained load over time"""
        logger.info("Running endurance test...")
        
        concurrent_users = min(20, self.config.max_concurrent_users // 2)
        duration = min(1800, self.config.duration_seconds)  # 30 minutes max
        
        return await self._execute_scenario(
            "endurance",
            concurrent_users,
            duration,
            self._normal_user_pattern
        )
    
    async def _execute_scenario(self, name: str, users: int, duration: int, user_pattern) -> Dict:
        """Execute a specific test scenario"""
        scenario_start = time.time()
        scenario_results = []
        
        # Create user tasks
        tasks = []
        for user_id in range(users):
            task = asyncio.create_task(
                self._simulate_user(f"{name}_user_{user_id}", duration, user_pattern)
            )
            tasks.append(task)
        
        # Wait for all users to complete
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for user_result in user_results:
            if isinstance(user_result, list):
                scenario_results.extend(user_result)
        
        scenario_duration = time.time() - scenario_start
        
        return {
            "scenario": name,
            "users": users,
            "duration": scenario_duration,
            "requests": len(scenario_results),
            "metrics": self._calculate_metrics(scenario_results),
            "raw_results": scenario_results,
        }
    
    async def _simulate_user(self, user_id: str, duration: int, pattern_func) -> List[TestResult]:
        """Simulate a single user's behavior"""
        user_results = []
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration:
                try:
                    # Get next request from user pattern
                    request_config = pattern_func()
                    
                    # Execute request
                    result = await self._make_request(session, user_id, request_config)
                    user_results.append(result)
                    
                    # Wait before next request (think time)
                    await asyncio.sleep(request_config.get("think_time", 1.0))
                    
                except Exception as e:
                    logger.error(f"User {user_id} error: {e}")
                    break
        
        return user_results
    
    async def _make_request(self, session: aiohttp.ClientSession, user_id: str, request_config: Dict) -> TestResult:
        """Make a single inference request and measure performance"""
        start_time = time.time()
        
        try:
            payload = {
                "model_name": request_config.get("model", "gpt-3.5-turbo"),
                "messages": request_config.get("messages", [
                    {"role": "user", "content": "Generate a creative story about AI."}
                ]),
                "max_tokens": request_config.get("max_tokens", 100),
                "temperature": request_config.get("temperature", 0.7),
                "user_id": user_id,
                "request_id": str(uuid.uuid4()),
            }
            
            async with session.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    response_data = await response.json()
                    tokens_generated = response_data.get("usage", {}).get("completion_tokens", 0)
                else:
                    tokens_generated = 0
                
                return TestResult(
                    timestamp=start_time,
                    latency_ms=latency_ms,
                    status_code=response.status,
                    tokens_generated=tokens_generated,
                    user_id=user_id,
                )
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return TestResult(
                timestamp=start_time,
                latency_ms=latency_ms,
                status_code=0,
                error_message=str(e),
                user_id=user_id,
            )
    
    def _normal_user_pattern(self) -> Dict:
        """Normal user behavior pattern"""
        patterns = [
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Write a short paragraph about renewable energy."}],
                "max_tokens": 150,
                "temperature": 0.7,
                "think_time": np.random.exponential(2.0),  # 2s average think time
            },
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
                "max_tokens": 200,
                "temperature": 0.5,
                "think_time": np.random.exponential(3.0),
            },
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Generate a Python function to sort a list."}],
                "max_tokens": 100,
                "temperature": 0.2,
                "think_time": np.random.exponential(1.5),
            },
        ]
        return np.random.choice(patterns)
    
    def _aggressive_user_pattern(self) -> Dict:
        """Aggressive user pattern for spike testing"""
        return {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Write a detailed analysis of market trends."}],
            "max_tokens": 500,
            "temperature": 0.8,
            "think_time": np.random.exponential(0.5),  # Very short think time
        }
    
    def _stress_user_pattern(self) -> Dict:
        """High-load pattern for stress testing"""
        return {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Quick response needed."}],
            "max_tokens": 50,
            "temperature": 0.1,
            "think_time": 0.1,  # Minimal think time
        }
    
    def _calculate_metrics(self, results: List[TestResult]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not results:
            return {}
        
        latencies = [r.latency_ms for r in results if r.status_code == 200]
        error_count = len([r for r in results if r.status_code != 200])
        total_tokens = sum(r.tokens_generated for r in results)
        
        if not latencies:
            return {"error_rate": 1.0, "total_requests": len(results)}
        
        return {
            "total_requests": len(results),
            "successful_requests": len(latencies),
            "error_count": error_count,
            "error_rate": error_count / len(results),
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "latency_std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "requests_per_second": len(results) / (max(r.timestamp for r in results) - min(r.timestamp for r in results)) if len(results) > 1 else 0,
            "tokens_per_second": total_tokens / (max(r.timestamp for r in results) - min(r.timestamp for r in results)) if len(results) > 1 else 0,
            "avg_tokens_per_request": total_tokens / len(latencies) if latencies else 0,
        }
    
    def _analyze_spike_behavior(self, spike_results: List[Dict]) -> Dict:
        """Analyze system behavior during spike testing"""
        if len(spike_results) < 2:
            return {}
        
        baseline_rps = spike_results[0]["metrics"].get("requests_per_second", 0)
        spike_rps = spike_results[1]["metrics"].get("requests_per_second", 0)
        recovery_rps = spike_results[2]["metrics"].get("requests_per_second", 0) if len(spike_results) > 2 else 0
        
        baseline_latency = spike_results[0]["metrics"].get("p95_latency_ms", 0)
        spike_latency = spike_results[1]["metrics"].get("p95_latency_ms", 0)
        
        return {
            "throughput_degradation": max(0, (baseline_rps - spike_rps) / baseline_rps) if baseline_rps > 0 else 0,
            "latency_increase": max(0, (spike_latency - baseline_latency) / baseline_latency) if baseline_latency > 0 else 0,
            "recovery_rate": max(0, recovery_rps / baseline_rps) if baseline_rps > 0 else 0,
            "spike_resilience_score": self._calculate_resilience_score(spike_results),
        }
    
    def _find_breaking_point(self, stress_results: List[Dict]) -> Dict:
        """Find the system's breaking point during stress testing"""
        breaking_point = None
        
        for i, result in enumerate(stress_results):
            metrics = result["metrics"]
            
            # Define breaking point criteria
            if (metrics.get("error_rate", 0) > 0.05 or  # 5% error rate
                metrics.get("p95_latency_ms", 0) > 5000 or  # 5s P95 latency
                metrics.get("requests_per_second", 0) < stress_results[0]["metrics"].get("requests_per_second", 0) * 0.5):  # 50% throughput drop
                
                breaking_point = {
                    "users_at_breaking_point": result["users"],
                    "error_rate": metrics.get("error_rate", 0),
                    "p95_latency_ms": metrics.get("p95_latency_ms", 0),
                    "throughput_degradation": 1 - (metrics.get("requests_per_second", 0) / stress_results[0]["metrics"].get("requests_per_second", 1)),
                }
                break
        
        return breaking_point or {"message": "No breaking point found within test limits"}
    
    def _calculate_resilience_score(self, results: List[Dict]) -> float:
        """Calculate overall system resilience score"""
        if len(results) < 2:
            return 0.0
        
        # Factors: error rate, latency stability, throughput maintenance
        error_penalty = sum(r["metrics"].get("error_rate", 0) for r in results) / len(results)
        
        latencies = [r["metrics"].get("p95_latency_ms", 0) for r in results]
        latency_stability = 1 - (statistics.stdev(latencies) / statistics.mean(latencies)) if statistics.mean(latencies) > 0 else 0
        
        throughputs = [r["metrics"].get("requests_per_second", 0) for r in results]
        throughput_stability = 1 - (statistics.stdev(throughputs) / statistics.mean(throughputs)) if statistics.mean(throughputs) > 0 else 0
        
        resilience_score = max(0, (latency_stability + throughput_stability) / 2 - error_penalty)
        return min(1.0, resilience_score)
    
    def _analyze_overall_results(self, all_results: List[Dict]) -> Dict:
        """Analyze overall test results across all scenarios"""
        summary = {
            "total_scenarios": len(all_results),
            "total_requests": sum(r.get("requests", 0) for r in all_results),
            "overall_success_rate": 0,
            "performance_grades": {},
            "recommendations": [],
        }
        
        # Calculate overall success rate
        total_successful = sum(r.get("metrics", {}).get("successful_requests", 0) for r in all_results)
        total_requests = sum(r.get("requests", 0) for r in all_results)
        
        if total_requests > 0:
            summary["overall_success_rate"] = total_successful / total_requests
        
        # Grade different aspects
        summary["performance_grades"] = {
            "latency": self._grade_latency_performance(all_results),
            "throughput": self._grade_throughput_performance(all_results),
            "reliability": self._grade_reliability(all_results),
            "scalability": self._grade_scalability(all_results),
        }
        
        # Generate recommendations
        summary["recommendations"] = self._generate_performance_recommendations(all_results)
        
        return summary
    
    def _grade_latency_performance(self, results: List[Dict]) -> str:
        """Grade latency performance"""
        avg_p95_latencies = [r.get("metrics", {}).get("p95_latency_ms", float('inf')) for r in results]
        avg_p95 = statistics.mean([l for l in avg_p95_latencies if l != float('inf')])
        
        if avg_p95 < 100:
            return "A+"
        elif avg_p95 < 200:
            return "A"
        elif avg_p95 < 500:
            return "B"
        elif avg_p95 < 1000:
            return "C"
        else:
            return "D"
    
    def _grade_throughput_performance(self, results: List[Dict]) -> str:
        """Grade throughput performance"""
        # Implementation based on requests per second relative to resource usage
        max_rps = max(r.get("metrics", {}).get("requests_per_second", 0) for r in results)
        
        if max_rps > 100:
            return "A+"
        elif max_rps > 50:
            return "A"
        elif max_rps > 25:
            return "B"
        elif max_rps > 10:
            return "C"
        else:
            return "D"
    
    def _grade_reliability(self, results: List[Dict]) -> str:
        """Grade system reliability"""
        avg_error_rate = statistics.mean([r.get("metrics", {}).get("error_rate", 1) for r in results])
        
        if avg_error_rate < 0.001:  # < 0.1%
            return "A+"
        elif avg_error_rate < 0.01:  # < 1%
            return "A"
        elif avg_error_rate < 0.05:  # < 5%
            return "B"
        elif avg_error_rate < 0.1:   # < 10%
            return "C"
        else:
            return "D"
    
    def _grade_scalability(self, results: List[Dict]) -> str:
        """Grade system scalability"""
        # Look for stress test results
        stress_results = [r for r in results if "stress" in r.get("scenario", "")]
        
        if not stress_results:
            return "N/A"
        
        # Check if performance degrades gracefully
        if len(stress_results) >= 3:
            early_rps = stress_results[0].get("metrics", {}).get("requests_per_second", 0)
            late_rps = stress_results[-1].get("metrics", {}).get("requests_per_second", 0)
            
            if late_rps > early_rps * 0.8:  # < 20% degradation
                return "A"
            elif late_rps > early_rps * 0.6:  # < 40% degradation
                return "B"
            else:
                return "C"
        
        return "B"  # Default for incomplete data
    
    def _generate_performance_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate actionable performance recommendations"""
        recommendations = []
        
        # Analyze patterns and suggest improvements
        avg_error_rate = statistics.mean([r.get("metrics", {}).get("error_rate", 0) for r in results])
        avg_latency = statistics.mean([r.get("metrics", {}).get("p95_latency_ms", 0) for r in results])
        max_rps = max(r.get("metrics", {}).get("requests_per_second", 0) for r in results)
        
        if avg_error_rate > 0.05:
            recommendations.append("🔴 High error rate detected. Consider implementing circuit breakers and better error handling.")
        
        if avg_latency > 1000:
            recommendations.append("🟡 High latency detected. Consider optimizing model inference, adding caching, or scaling compute resources.")
        
        if max_rps < 50:
            recommendations.append("🟡 Low throughput detected. Consider optimizing batch processing, increasing concurrency, or scaling horizontally.")
        
        # Check for specific patterns
        spike_results = [r for r in results if "spike" in r.get("scenario", "")]
        if spike_results and any(r.get("metrics", {}).get("error_rate", 0) > 0.1 for r in spike_results):
            recommendations.append("🔴 Poor spike handling. Implement auto-scaling and load balancing improvements.")
        
        stress_results = [r for r in results if "stress" in r.get("scenario", "")]
        if stress_results:
            breaking_point = self._find_breaking_point(stress_results)
            if "users_at_breaking_point" in breaking_point and breaking_point["users_at_breaking_point"] < 50:
                recommendations.append("🔴 Low scalability limit. Consider architectural improvements for better horizontal scaling.")
        
        if not recommendations:
            recommendations.append("✅ System performance is within acceptable ranges. Continue monitoring and consider capacity planning for growth.")
        
        return recommendations
    
    async def _generate_load_test_report(self, results: Dict):
        """Generate comprehensive HTML load test report"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Inference Platform - Load Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .grade-A {{ color: #27ae60; }}
                .grade-B {{ color: #f39c12; }}
                .grade-C {{ color: #e67e22; }}
                .grade-D {{ color: #e74c3c; }}
                .recommendations {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart-container {{ margin: 20px 0; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>AI Inference Platform Load Test Report</h1>
                <p><strong>Test Duration:</strong> {results['config']['duration_seconds']}s</p>
                <p><strong>Max Concurrent Users:</strong> {results['config']['max_concurrent_users']}</p>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                {self._generate_summary_cards(results['summary'])}
            </div>
            
            <div class="recommendations">
                <h3>📋 Performance Recommendations</h3>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in results['summary']['recommendations'])}
                </ul>
            </div>
            
            <div class="chart-container">
                <h3>Performance Metrics Over Time</h3>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
            
            <h3>📈 Detailed Scenario Results</h3>
            {self._generate_scenario_tables(results['results'])}
            
            <script>
                {self._generate_chart_script(results['results'])}
            </script>
        </body>
        </html>
        """
        
        # Save report
        report_filename = f"load_test_report_{int(time.time())}.html"
        with open(report_filename, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Load test report generated: {report_filename}")
    
    def _generate_summary_cards(self, summary: Dict) -> str:
        """Generate HTML summary cards"""
        grades = summary.get('performance_grades', {})
        
        cards = []
        for metric, grade in grades.items():
            card = f"""
            <div class="metric-card">
                <div class="metric-value grade-{grade.replace('+', '')}">{grade}</div>
                <div>{metric.title()} Performance</div>
            </div>
            """
            cards.append(card)
        
        # Add overall success rate card
        success_rate = summary.get('overall_success_rate', 0)
        cards.append(f"""
        <div class="metric-card">
            <div class="metric-value">{success_rate:.1%}</div>
            <div>Overall Success Rate</div>
        </div>
        """)
        
        return ''.join(cards)
    
    def _generate_scenario_tables(self, results: List[Dict]) -> str:
        """Generate HTML tables for scenario results"""
        tables = []
        
        for result in results:
            metrics = result.get('metrics', {})
            
            table = f"""
            <h4>{result['scenario'].replace('_', ' ').title()}</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Requests</td><td>{metrics.get('total_requests', 0):,}</td></tr>
                <tr><td>Success Rate</td><td>{(1 - metrics.get('error_rate', 0)):.1%}</td></tr>
                <tr><td>Avg Latency</td><td>{metrics.get('avg_latency_ms', 0):.1f} ms</td></tr>
                <tr><td>P95 Latency</td><td>{metrics.get('p95_latency_ms', 0):.1f} ms</td></tr>
                <tr><td>P99 Latency</td><td>{metrics.get('p99_latency_ms', 0):.1f} ms</td></tr>
                <tr><td>Requests/sec</td><td>{metrics.get('requests_per_second', 0):.1f}</td></tr>
                <tr><td>Tokens/sec</td><td>{metrics.get('tokens_per_second', 0):.1f}</td></tr>
            </table>
            """
            tables.append(table)
        
        return ''.join(tables)
    
    def _generate_chart_script(self, results: List[Dict]) -> str:
        """Generate JavaScript for performance charts"""
        # Extract data for charts
        scenarios = [r['scenario'] for r in results]
        latencies = [r.get('metrics', {}).get('p95_latency_ms', 0) for r in results]
        throughputs = [r.get('metrics', {}).get('requests_per_second', 0) for r in results]
        
        return f"""
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(scenarios)},
                datasets: [{{
                    label: 'P95 Latency (ms)',
                    data: {json.dumps(latencies)},
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1,
                    yAxisID: 'y'
                }}, {{
                    label: 'Requests/sec',
                    data: {json.dumps(throughputs)},
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1,
                    yAxisID: 'y1'
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{ display: true, text: 'Latency (ms)' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{ display: true, text: 'Requests/sec' }},
                        grid: {{ drawOnChartArea: false }}
                    }}
                }}
            }}
        }});
        """


async def main():
    """Main load testing execution"""
    parser = argparse.ArgumentParser(description='AI Inference Platform Load Testing')
    parser.add_argument('--url', default='http://localhost:8080', help='Base URL')
    parser.add_argument('--users', type=int, default=50, help='Max concurrent users')
    parser.add_argument('--duration', type=int, default=600, help='Test duration (seconds)')
    parser.add_argument('--ramp-up', type=int, default=60, help='Ramp-up time (seconds)')
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        base_url=args.url,
        duration_seconds=args.duration,
        ramp_up_seconds=args.ramp_up,
        max_concurrent_users=args.users,
        request_patterns=[],
    )
    
    runner = LoadTestRunner(config)
    results = await runner.run_load_test()
    
    print("\n" + "="*60)
    print("LOAD TEST COMPLETED")
    print("="*60)
    print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.1%}")
    print(f"Performance Grades:")
    for metric, grade in results['summary']['performance_grades'].items():
        print(f"  {metric.title()}: {grade}")
    print("\nTop Recommendations:")
    for rec in results['summary']['recommendations'][:3]:
        print(f"  • {rec}")


if __name__ == "__main__":
    asyncio.run(main())