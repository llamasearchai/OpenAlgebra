#!/usr/bin/env python3
"""
Generate consolidated report from multiple load test runs
"""

import json
import argparse
from pathlib import Path
import glob
import re
from datetime import datetime
import statistics
from typing import Dict, List, Any
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsolidatedReportGenerator:
    """Generate unified reports from multiple medical AI load test runs"""
    
    def __init__(self, report_dir: Path, environment: str, config_file: Path):
        self.report_dir = Path(report_dir)
        self.environment = environment
        self.config_file = Path(config_file)
        self.test_results = []
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load test configuration"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def collect_test_results(self) -> List[Dict]:
        """Collect all test results from the report directory"""
        results = []
        
        # Find all HTML reports
        report_files = list(self.report_dir.glob("*.html"))
        log_files = list(self.report_dir.glob("*.log"))
        
        logger.info(f"Found {len(report_files)} HTML reports and {len(log_files)} log files")
        
        # Extract data from log files
        for log_file in log_files:
            test_type = self._extract_test_type(log_file.name)
            result_data = self._parse_log_file(log_file)
            if result_data:
                result_data['test_type'] = test_type
                result_data['log_file'] = str(log_file)
                results.append(result_data)
        
        return results
    
    def _extract_test_type(self, filename: str) -> str:
        """Extract test type from filename"""
        if 'baseline' in filename:
            return 'baseline'
        elif 'compliance' in filename:
            return 'compliance'
        elif 'emergency' in filename:
            return 'emergency'
        elif 'stress' in filename:
            return 'stress'
        elif 'federated' in filename:
            return 'federated'
        else:
            return 'unknown'
    
    def _parse_log_file(self, log_file: Path) -> Dict:
        """Parse performance metrics from log file"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract key metrics using regex
            metrics = {}
            
            # Success rate
            success_match = re.search(r'Overall Success Rate: ([\d.]+)%', content)
            if success_match:
                metrics['success_rate'] = float(success_match.group(1)) / 100
            
            # Clinical safety score
            safety_match = re.search(r'Clinical Safety Score: ([\d.]+)', content)
            if safety_match:
                metrics['clinical_safety_score'] = float(safety_match.group(1))
            
            # HIPAA compliance
            hipaa_match = re.search(r'HIPAA Compliance Rate: ([\d.]+)%', content)
            if hipaa_match:
                metrics['hipaa_compliance_rate'] = float(hipaa_match.group(1)) / 100
            
            # Performance grades
            grades = {}
            grade_pattern = r'(\w+): ([A-D][+]?)'
            for match in re.finditer(grade_pattern, content):
                grades[match.group(1).lower()] = match.group(2)
            metrics['performance_grades'] = grades
            
            # Risk level
            risk_match = re.search(r'Overall Risk Level: (\w+) \(([\d.]+)\)', content)
            if risk_match:
                metrics['risk_level'] = risk_match.group(1)
                metrics['risk_score'] = float(risk_match.group(2))
            
            # Extract recommendations
            recommendations = []
            rec_section = re.search(r'Top Clinical Recommendations:(.*?)(?:\n\n|\Z)', content, re.DOTALL)
            if rec_section:
                for line in rec_section.group(1).split('\n'):
                    if line.strip().startswith('•'):
                        recommendations.append(line.strip()[1:].strip())
            metrics['recommendations'] = recommendations
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to parse log file {log_file}: {e}")
            return {}
    
    def generate_consolidated_report(self) -> str:
        """Generate comprehensive consolidated HTML report"""
        self.test_results = self.collect_test_results()
        
        overall_analysis = self._analyze_overall_performance()
        clinical_readiness = self._assess_clinical_readiness()
        compliance_summary = self._summarize_compliance_status()
        recommendations = self._generate_consolidated_recommendations()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical AI System - Consolidated Load Test Report</title>
            <style>
                {self._get_report_styles()}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>Medical AI System - Consolidated Load Test Report</h1>
                <div class="header-info">
                    <div><strong>Environment:</strong> {self.environment.title()}</div>
                    <div><strong>Test Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <div><strong>Test Scenarios:</strong> {len(self.test_results)}</div>
                </div>
            </div>

            <div class="executive-summary">
                <h2>Executive Summary</h2>
                {self._generate_executive_summary_html(overall_analysis, clinical_readiness)}
            </div>

            <div class="clinical-readiness">
                <h2>Clinical Deployment Readiness</h2>
                {self._generate_clinical_readiness_html(clinical_readiness)}
            </div>

            <div class="compliance-status">
                <h2>Regulatory Compliance Status</h2>
                {self._generate_compliance_status_html(compliance_summary)}
            </div>

            <div class="performance-overview">
                <h2>Performance Overview</h2>
                {self._generate_performance_overview_html(overall_analysis)}
            </div>

            <div class="test-scenarios">
                <h2>🧪 Test Scenario Results</h2>
                {self._generate_scenario_results_html()}
            </div>

            <div class="recommendations">
                <h2>Recommendations & Action Items</h2>
                {self._generate_recommendations_html(recommendations)}
            </div>

            <div class="charts">
                <h2>Performance Trends</h2>
                <canvas id="performanceTrends" width="400" height="200"></canvas>
                <canvas id="complianceMetrics" width="400" height="200"></canvas>
            </div>

            <script>
                {self._generate_chart_scripts()}
            </script>
        </body>
        </html>
        """
        
        # Save consolidated report
        report_file = self.report_dir / "consolidated_report.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Consolidated report generated: {report_file}")
        return str(report_file)
    
    def _analyze_overall_performance(self) -> Dict:
        """Analyze overall system performance across all tests"""
        if not self.test_results:
            return {}
        
        # Aggregate metrics
        success_rates = [r.get('success_rate', 0) for r in self.test_results if 'success_rate' in r]
        safety_scores = [r.get('clinical_safety_score', 0) for r in self.test_results if 'clinical_safety_score' in r]
        risk_scores = [r.get('risk_score', 0) for r in self.test_results if 'risk_score' in r]
        
        analysis = {
            'overall_success_rate': statistics.mean(success_rates) if success_rates else 0,
            'min_success_rate': min(success_rates) if success_rates else 0,
            'max_success_rate': max(success_rates) if success_rates else 0,
            'avg_clinical_safety_score': statistics.mean(safety_scores) if safety_scores else 0,
            'avg_risk_score': statistics.mean(risk_scores) if risk_scores else 0,
            'consistency_score': 1 - (statistics.stdev(success_rates) if len(success_rates) > 1 else 0),
            'test_coverage': len(self.test_results)
        }
        
        # Overall grade calculation
        if analysis['overall_success_rate'] >= 0.99 and analysis['avg_risk_score'] < 0.2:
            analysis['overall_grade'] = 'A+'
        elif analysis['overall_success_rate'] >= 0.95 and analysis['avg_risk_score'] < 0.4:
            analysis['overall_grade'] = 'A'
        elif analysis['overall_success_rate'] >= 0.90 and analysis['avg_risk_score'] < 0.6:
            analysis['overall_grade'] = 'B'
        elif analysis['overall_success_rate'] >= 0.85:
            analysis['overall_grade'] = 'C'
        else:
            analysis['overall_grade'] = 'D'
        
        return analysis
    
    def _assess_clinical_readiness(self) -> Dict:
        """Assess clinical deployment readiness"""
        readiness = {
            'deployment_ready': False,
            'readiness_score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'approval_criteria': {}
        }
        
        # Check each test result for clinical readiness criteria
        criteria_scores = []
        
        for result in self.test_results:
            test_score = 0.0
            
            # Success rate criteria (30% weight)
            success_rate = result.get('success_rate', 0)
            if success_rate >= 0.99:
                test_score += 0.3
            elif success_rate >= 0.95:
                test_score += 0.25
            elif success_rate >= 0.90:
                test_score += 0.15
            else:
                readiness['critical_issues'].append(f"Low success rate in {result.get('test_type', 'unknown')} test: {success_rate:.1%}")
            
            # Clinical safety score (25% weight)
            safety_score = result.get('clinical_safety_score', 0)
            if safety_score >= 0.9:
                test_score += 0.25
            elif safety_score >= 0.8:
                test_score += 0.20
            elif safety_score >= 0.7:
                test_score += 0.10
            else:
                readiness['critical_issues'].append(f"Low clinical safety score in {result.get('test_type', 'unknown')} test: {safety_score:.2f}")
            
            # HIPAA compliance (25% weight)
            hipaa_rate = result.get('hipaa_compliance_rate', 0)
            if hipaa_rate >= 0.99:
                test_score += 0.25
            elif hipaa_rate >= 0.95:
                test_score += 0.15
            else:
                readiness['critical_issues'].append(f"HIPAA compliance issues in {result.get('test_type', 'unknown')} test: {hipaa_rate:.1%}")
            
            # Risk assessment (20% weight)
            risk_score = result.get('risk_score', 1.0)
            if risk_score < 0.2:
                test_score += 0.20
            elif risk_score < 0.4:
                test_score += 0.15
            elif risk_score < 0.6:
                test_score += 0.10
            else:
                readiness['warnings'].append(f"High risk score in {result.get('test_type', 'unknown')} test: {risk_score:.2f}")
            
            criteria_scores.append(test_score)
        
        # Calculate overall readiness score
        readiness['readiness_score'] = statistics.mean(criteria_scores) if criteria_scores else 0.0
        
        # Determine deployment readiness
        if readiness['readiness_score'] >= 0.85 and not readiness['critical_issues']:
            readiness['deployment_ready'] = True
            readiness['status'] = 'APPROVED'
        elif readiness['readiness_score'] >= 0.70 and len(readiness['critical_issues']) <= 1:
            readiness['status'] = 'CONDITIONAL'
        else:
            readiness['status'] = 'NOT_APPROVED'
        
        return readiness
    
    def _summarize_compliance_status(self) -> Dict:
        """Summarize regulatory compliance status"""
        compliance = {
            'hipaa_status': 'UNKNOWN',
            'fda_status': 'UNKNOWN',
            'gdpr_status': 'UNKNOWN',
            'overall_compliance': 'UNKNOWN',
            'compliance_score': 0.0,
            'violations': []
        }
        
        # Aggregate HIPAA compliance rates
        hipaa_rates = [r.get('hipaa_compliance_rate', 0) for r in self.test_results if 'hipaa_compliance_rate' in r]
        
        if hipaa_rates:
            avg_hipaa = statistics.mean(hipaa_rates)
            min_hipaa = min(hipaa_rates)
            
            if min_hipaa >= 0.99:
                compliance['hipaa_status'] = 'COMPLIANT'
            elif avg_hipaa >= 0.95:
                compliance['hipaa_status'] = 'MOSTLY_COMPLIANT'
            else:
                compliance['hipaa_status'] = 'NON_COMPLIANT'
                compliance['violations'].append(f"HIPAA compliance rate below threshold: {avg_hipaa:.1%}")
        
        # Check for specific compliance requirements from config
        if self.config.get('compliance_requirements'):
            requirements = self.config['compliance_requirements']
            
            # FDA validation requirements
            if requirements.get('fda_requirements'):
                # This would be checked against actual test results
                compliance['fda_status'] = 'PENDING_VALIDATION'
            
            # GDPR requirements
            if requirements.get('gdpr'):
                # This would be checked against data handling practices
                compliance['gdpr_status'] = 'PENDING_VALIDATION'
        
        # Calculate overall compliance score
        status_scores = {
            'COMPLIANT': 1.0,
            'MOSTLY_COMPLIANT': 0.7,
            'PENDING_VALIDATION': 0.5,
            'NON_COMPLIANT': 0.0,
            'UNKNOWN': 0.3
        }
        
        scores = [
            status_scores.get(compliance['hipaa_status'], 0),
            status_scores.get(compliance['fda_status'], 0),
            status_scores.get(compliance['gdpr_status'], 0)
        ]
        
        compliance['compliance_score'] = statistics.mean(scores)
        
        if compliance['compliance_score'] >= 0.9:
            compliance['overall_compliance'] = 'FULLY_COMPLIANT'
        elif compliance['compliance_score'] >= 0.7:
            compliance['overall_compliance'] = 'MOSTLY_COMPLIANT'
        elif compliance['compliance_score'] >= 0.5:
            compliance['overall_compliance'] = 'PARTIAL_COMPLIANCE'
        else:
            compliance['overall_compliance'] = 'NON_COMPLIANT'
        
        return compliance
    
    def _generate_consolidated_recommendations(self) -> List[Dict]:
        """Generate prioritized recommendations from all test results"""
        all_recommendations = []
        
        # Collect recommendations from all tests
        for result in self.test_results:
            test_type = result.get('test_type', 'unknown')
            for rec in result.get('recommendations', []):
                all_recommendations.append({
                    'recommendation': rec,
                    'test_type': test_type,
                    'priority': self._determine_priority(rec),
                    'category': self._categorize_recommendation(rec)
                })
        
        # Group and prioritize
        categorized = {}
        for rec in all_recommendations:
            category = rec['category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(rec)
        
        # Sort by priority within categories
        for category in categorized:
            categorized[category].sort(key=lambda x: x['priority'], reverse=True)
        
        return categorized
    
    def _determine_priority(self, recommendation: str) -> int:
        """Determine priority level of recommendation (1-5, 5 being highest)"""
        if 'CRITICAL' in recommendation or 'critical' in recommendation:
            return 5
        elif 'WARNING' in recommendation or 'High error rate' in recommendation:
            return 4
        elif 'CAUTION' in recommendation or 'Consider' in recommendation:
            return 3
        elif 'Monitor' in recommendation or 'optimize' in recommendation:
            return 2
        else:
            return 1
    
    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize recommendation by type"""
        if any(term in recommendation.lower() for term in ['compliance', 'hipaa', 'gdpr', 'fda']):
            return 'regulatory_compliance'
        elif any(term in recommendation.lower() for term in ['latency', 'performance', 'throughput']):
            return 'performance_optimization'
        elif any(term in recommendation.lower() for term in ['error', 'reliability', 'availability']):
            return 'reliability_improvement'
        elif any(term in recommendation.lower() for term in ['scaling', 'capacity', 'load']):
            return 'scalability_enhancement'
        elif any(term in recommendation.lower() for term in ['clinical', 'safety', 'accuracy']):
            return 'clinical_safety'
        else:
            return 'general_improvement'
    
    def _get_report_styles(self) -> str:
        """Get CSS styles for the consolidated report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .header-info {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
        }
        
        .header-info div {
            background: rgba(255,255,255,0.1);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
        }
        
        .section {
            background: white;
            margin: 1rem auto;
            padding: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 1200px;
        }
        
        .executive-summary, .clinical-readiness, .compliance-status, 
        .performance-overview, .test-scenarios, .recommendations, .charts {
            background: white;
            margin: 1rem auto;
            padding: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 1200px;
        }
        
        h2 {
            color: #2c3e50;
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .status-card {
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            font-weight: bold;
        }
        
        .status-approved {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-conditional {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status-not-approved {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .metric-label {
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        .grade-A { color: #27ae60; }
        .grade-B { color: #f39c12; }
        .grade-C { color: #e67e22; }
        .grade-D { color: #e74c3c; }
        
        .recommendations-category {
            margin-bottom: 2rem;
        }
        
        .recommendations-category h3 {
            color: #34495e;
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: #ecf0f1;
            border-radius: 0.25rem;
        }
        
        .recommendation-item {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #3498db;
            background: #f8f9fa;
            border-radius: 0.25rem;
        }
        
        .priority-5 { border-left-color: #e74c3c; }
        .priority-4 { border-left-color: #f39c12; }
        .priority-3 { border-left-color: #f1c40f; }
        .priority-2 { border-left-color: #3498db; }
        .priority-1 { border-left-color: #95a5a6; }
        
        .scenario-result {
            margin-bottom: 1.5rem;
            padding: 1rem;
            border: 1px solid #dee2e6;
            border-radius: 0.5rem;
        }
        
        .scenario-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .scenario-title {
            font-size: 1.2rem;
            font-weight: bold;
            text-transform: capitalize;
        }
        
        .scenario-status {
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.85rem;
            font-weight: bold;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }
        
        .chart-container {
            margin: 1rem 0;
            padding: 1rem;
            background: white;
            border-radius: 0.5rem;
        }
        
        .alert {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .alert-warning {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .alert-danger {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        """
    
    def _generate_executive_summary_html(self, analysis: Dict, readiness: Dict) -> str:
        """Generate executive summary HTML"""
        status_class = {
            'APPROVED': 'status-approved',
            'CONDITIONAL': 'status-conditional',
            'NOT_APPROVED': 'status-not-approved'
        }.get(readiness.get('status', 'NOT_APPROVED'), 'status-not-approved')
        
        return f"""
        <div class="status-grid">
            <div class="status-card {status_class}">
                <h3>Deployment Status</h3>
                <div style="font-size: 1.5rem; margin: 0.5rem 0;">
                    {readiness.get('status', 'NOT_APPROVED')}
                </div>
                <div>Readiness Score: {readiness.get('readiness_score', 0):.1%}</div>
            </div>
            
            <div class="status-card">
                <h3>Overall Performance</h3>
                <div style="font-size: 1.5rem; margin: 0.5rem 0;" class="grade-{analysis.get('overall_grade', 'D').replace('+', '')}">
                    {analysis.get('overall_grade', 'D')}
                </div>
                <div>Success Rate: {analysis.get('overall_success_rate', 0):.1%}</div>
            </div>
            
            <div class="status-card">
                <h3>Clinical Safety</h3>
                <div style="font-size: 1.5rem; margin: 0.5rem 0;">
                    {analysis.get('avg_clinical_safety_score', 0):.2f}
                </div>
                <div>Risk Score: {analysis.get('avg_risk_score', 0):.2f}</div>
            </div>
            
            <div class="status-card">
                <h3>Test Coverage</h3>
                <div style="font-size: 1.5rem; margin: 0.5rem 0;">
                    {analysis.get('test_coverage', 0)}
                </div>
                <div>Scenarios Tested</div>
            </div>
        </div>
        
        <div class="alert {'alert-success' if readiness.get('deployment_ready') else 'alert-warning'}">
            <strong>Summary:</strong> 
            {'System is ready for clinical deployment with appropriate monitoring.' if readiness.get('deployment_ready') 
             else 'System requires additional validation before clinical deployment.'}
        </div>
        """
    
    def _generate_clinical_readiness_html(self, readiness: Dict) -> str:
        """Generate clinical readiness assessment HTML"""
        critical_issues = readiness.get('critical_issues', [])
        warnings = readiness.get('warnings', [])
        
        issues_html = ""
        if critical_issues:
            issues_html += "<div class='alert alert-danger'><h4>Critical Issues:</h4><ul>"
            for issue in critical_issues:
                issues_html += f"<li>{issue}</li>"
            issues_html += "</ul></div>"
        
        if warnings:
            issues_html += "<div class='alert alert-warning'><h4>Warnings:</h4><ul>"
            for warning in warnings:
                issues_html += f"<li>{warning}</li>"
            issues_html += "</ul></div>"
        
        if not critical_issues and not warnings:
            issues_html = "<div class='alert alert-success'>No critical issues or warnings identified.</div>"
        
        return f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{readiness.get('readiness_score', 0):.1%}</div>
                <div class="metric-label">Readiness Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(critical_issues)}</div>
                <div class="metric-label">Critical Issues</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(warnings)}</div>
                <div class="metric-label">Warnings</div>
            </div>
        </div>
        
        {issues_html}
        """

def generate_report(report_dir, environment, config_path):
    # Load test data
    test_data = []
    for test_file in Path(report_dir).glob('*.log'):
        with open(test_file) as f:
            test_data.append(parse_log(f.read()))
    
    # Generate metrics
    df = pd.DataFrame(test_data)
    metrics = {
        'environment': environment,
        'timestamp': datetime.utcnow().isoformat(),
        'success_rate': df['success'].mean(),
        'avg_latency': df['latency'].mean(),
        'max_latency': df['latency'].max()
    }
    
    # Save report
    report_path = Path(report_dir) / "consolidated_report.html"
    report_path.write_text(generate_html(metrics, config_path))

def generate_html(metrics, config_path):
    return f"""
    <html>
    <body>
        <h1>Medical AI Load Test Report</h1>
        <p>Environment: {metrics['environment']}</p>
        <p>Success Rate: {metrics['success_rate']:.2%}</p>
        <p>Average Latency: {metrics['avg_latency']:.2f}s</p>
        <p>Max Latency: {metrics['max_latency']:.2f}s</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate_report(args.report_dir, args.environment, args.config)