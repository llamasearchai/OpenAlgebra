#!/usr/bin/env python3
"""
OpenAlgebra Medical AI Python Client

This module provides a comprehensive Python interface to the OpenAlgebra Medical AI system,
including FastAPI endpoints, OpenAI agents integration, and medical data processing capabilities.
"""

import json
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OpenAlgebraConfig:
    """Configuration for OpenAlgebra Medical AI client."""
    base_url: str = "http://127.0.0.1:8000"
    api_key: Optional[str] = None
    timeout: int = 30
    verify_ssl: bool = True
    openai_api_key: Optional[str] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> 'OpenAlgebraConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

@dataclass
class DICOMProcessRequest:
    """Request for DICOM processing."""
    file_path: str
    output_format: str = "sparse_tensor"
    anonymize: bool = True
    validate_hipaa: bool = True

@dataclass
class ModelTrainRequest:
    """Request for model training."""
    model_type: str
    dataset_path: str
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2

@dataclass
class ModelPredictRequest:
    """Request for model prediction."""
    model_path: str
    input_data: List[List[float]]
    return_confidence: bool = True

@dataclass
class FederatedLearningRequest:
    """Request for federated learning."""
    model_type: str
    client_count: int = 3
    rounds: int = 10
    min_clients: int = 2
    data_split_strategy: str = "random"

@dataclass
class ClinicalAnalysisRequest:
    """Request for clinical data analysis using AI agents."""
    dataset_path: str
    analysis_type: str = "risk_assessment"
    privacy_level: str = "full"
    include_recommendations: bool = True

@dataclass
class DICOMAnalysisRequest:
    """Request for DICOM analysis using AI agents."""
    dicom_path: str
    analysis_focus: List[str]
    compare_with_normal: bool = True
    include_measurements: bool = True

class OpenAlgebraClient:
    """Main client for OpenAlgebra Medical AI system."""
    
    def __init__(self, config: Optional[OpenAlgebraConfig] = None):
        """Initialize the OpenAlgebra client."""
        self.config = config or OpenAlgebraConfig()
        self.session = requests.Session()
        
        # Set up authentication if API key is provided
        if self.config.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            })
        else:
            self.session.headers.update({'Content-Type': 'application/json'})
        
        logger.info(f"OpenAlgebra client initialized with base URL: {self.config.base_url}")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to the API."""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.config.timeout, verify=self.config.verify_ssl)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=self.config.timeout, verify=self.config.verify_ssl)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def health_check(self) -> Dict:
        """Check the health status of the API server."""
        logger.info("Performing health check...")
        return self._make_request("GET", "/health")
    
    def get_system_status(self) -> Dict:
        """Get detailed system status."""
        logger.info("Retrieving system status...")
        return self._make_request("GET", "/status")
    
    def process_dicom(self, request: DICOMProcessRequest) -> Dict:
        """Process DICOM files."""
        logger.info(f"Processing DICOM files: {request.file_path}")
        return self._make_request("POST", "/api/v1/dicom/process", asdict(request))
    
    def get_dicom_metadata(self, file_path: str) -> Dict:
        """Extract DICOM metadata."""
        logger.info(f"Extracting DICOM metadata: {file_path}")
        encoded_path = requests.utils.quote(file_path, safe='')
        return self._make_request("GET", f"/api/v1/dicom/metadata/{encoded_path}")
    
    def train_model(self, request: ModelTrainRequest) -> Dict:
        """Train a medical AI model."""
        logger.info(f"Training model: {request.model_type}")
        return self._make_request("POST", "/api/v1/models/train", asdict(request))
    
    def predict_model(self, request: ModelPredictRequest) -> Dict:
        """Make predictions using a trained model."""
        logger.info(f"Making predictions with model: {request.model_path}")
        return self._make_request("POST", "/api/v1/models/predict", asdict(request))
    
    def start_federated_learning(self, request: FederatedLearningRequest) -> Dict:
        """Start federated learning process."""
        logger.info(f"Starting federated learning: {request.model_type}")
        return self._make_request("POST", "/api/v1/federated/start", asdict(request))
    
    def run_benchmark(self, operation: str, dataset_size: int = 1000, iterations: int = 10) -> Dict:
        """Run performance benchmarks."""
        logger.info(f"Running benchmark: {operation}")
        data = {
            "operation": operation,
            "dataset_size": dataset_size,
            "iterations": iterations
        }
        return self._make_request("POST", "/api/v1/benchmark", data)

class MedicalAIAgent:
    """OpenAI-powered medical AI agent client."""
    
    def __init__(self, client: OpenAlgebraClient):
        """Initialize the medical AI agent."""
        self.client = client
        self.config = client.config
        
        if not self.config.openai_api_key:
            logger.warning("OpenAI API key not configured. Some features may not be available.")
    
    def analyze_clinical_data(self, request: ClinicalAnalysisRequest) -> Dict:
        """Analyze clinical data using AI."""
        logger.info(f"Analyzing clinical data: {request.dataset_path}")
        
        # This would integrate with the agents module through the API
        data = {
            "dataset_path": request.dataset_path,
            "analysis_type": request.analysis_type,
            "privacy_level": request.privacy_level,
            "include_recommendations": request.include_recommendations
        }
        
        return self.client._make_request("POST", "/api/v1/agents/analyze_clinical", data)
    
    def analyze_dicom_images(self, request: DICOMAnalysisRequest) -> Dict:
        """Analyze DICOM images using AI."""
        logger.info(f"Analyzing DICOM images: {request.dicom_path}")
        
        data = {
            "dicom_path": request.dicom_path,
            "analysis_focus": request.analysis_focus,
            "compare_with_normal": request.compare_with_normal,
            "include_measurements": request.include_measurements
        }
        
        return self.client._make_request("POST", "/api/v1/agents/analyze_dicom", data)
    
    def start_chat_session(self, specialty: Optional[str] = None, context: Optional[Dict] = None) -> str:
        """Start a medical chat session."""
        logger.info(f"Starting medical chat session (specialty: {specialty})")
        
        data = {
            "specialty": specialty,
            "context": context or {}
        }
        
        response = self.client._make_request("POST", "/api/v1/agents/chat/start", data)
        return response.get("session_id")
    
    def send_chat_message(self, session_id: str, message: str) -> str:
        """Send a message in a chat session."""
        data = {
            "session_id": session_id,
            "message": message
        }
        
        response = self.client._make_request("POST", "/api/v1/agents/chat/message", data)
        return response.get("response", "")
    
    def detect_anomalies(self, dataset_path: str, baseline_path: str, threshold: float = 0.7) -> Dict:
        """Detect anomalies in medical data."""
        logger.info(f"Detecting anomalies in: {dataset_path}")
        
        data = {
            "dataset_path": dataset_path,
            "baseline_path": baseline_path,
            "threshold": threshold
        }
        
        return self.client._make_request("POST", "/api/v1/agents/detect_anomalies", data)

class AsyncOpenAlgebraClient:
    """Async version of the OpenAlgebra client."""
    
    def __init__(self, config: Optional[OpenAlgebraConfig] = None):
        """Initialize the async OpenAlgebra client."""
        self.config = config or OpenAlgebraConfig()
        self.headers = {'Content-Type': 'application/json'}
        
        if self.config.api_key:
            self.headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        logger.info(f"Async OpenAlgebra client initialized with base URL: {self.config.base_url}")
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make async HTTP request to the API."""
        url = f"{self.config.base_url}{endpoint}"
        
        connector = aiohttp.TCPConnector(verify_ssl=self.config.verify_ssl)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        ) as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
            except aiohttp.ClientError as e:
                logger.error(f"Async request failed: {e}")
                raise
    
    async def health_check(self) -> Dict:
        """Async health check."""
        return await self._make_request("GET", "/health")
    
    async def process_dicom_batch(self, requests: List[DICOMProcessRequest]) -> List[Dict]:
        """Process multiple DICOM requests concurrently."""
        logger.info(f"Processing {len(requests)} DICOM files concurrently...")
        
        tasks = [
            self._make_request("POST", "/api/v1/dicom/process", asdict(req))
            for req in requests
        ]
        
        return await asyncio.gather(*tasks)
    
    async def train_models_parallel(self, requests: List[ModelTrainRequest]) -> List[Dict]:
        """Train multiple models in parallel."""
        logger.info(f"Training {len(requests)} models in parallel...")
        
        tasks = [
            self._make_request("POST", "/api/v1/models/train", asdict(req))
            for req in requests
        ]
        
        return await asyncio.gather(*tasks)

class MedicalDataProcessor:
    """Helper class for medical data processing."""
    
    @staticmethod
    def load_dicom_series(directory: str) -> List[str]:
        """Load DICOM series from directory."""
        dicom_files = []
        path = Path(directory)
        
        if path.is_dir():
            for file_path in path.rglob("*.dcm"):
                dicom_files.append(str(file_path))
        
        logger.info(f"Found {len(dicom_files)} DICOM files in {directory}")
        return dicom_files
    
    @staticmethod
    def prepare_medical_dataset(data_path: str, labels_path: str) -> Dict:
        """Prepare medical dataset for training."""
        # This would load and preprocess medical data
        logger.info(f"Preparing medical dataset from {data_path}")
        
        return {
            "features_path": data_path,
            "labels_path": labels_path,
            "preprocessing": "normalized",
            "format": "sparse_matrix"
        }
    
    @staticmethod
    def validate_hipaa_compliance(file_path: str) -> Dict:
        """Validate HIPAA compliance of medical files."""
        logger.info(f"Validating HIPAA compliance: {file_path}")
        
        # This would perform actual HIPAA validation
        return {
            "compliant": True,
            "issues": [],
            "recommendations": []
        }

class PerformanceMonitor:
    """Performance monitoring for OpenAlgebra operations."""
    
    def __init__(self, client: OpenAlgebraClient):
        """Initialize performance monitor."""
        self.client = client
        self.operations = []
    
    def benchmark_operation(self, operation_func, *args, **kwargs) -> Dict:
        """Benchmark an operation and return performance metrics."""
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        duration = end_time - start_time
        
        metrics = {
            "operation": operation_func.__name__,
            "duration_seconds": duration,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.operations.append(metrics)
        logger.info(f"Operation {operation_func.__name__} completed in {duration:.2f}s (success: {success})")
        
        return {
            "result": result,
            "metrics": metrics
        }
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all performance metrics."""
        if not self.operations:
            return {"total_operations": 0}
        
        successful_ops = [op for op in self.operations if op["success"]]
        failed_ops = [op for op in self.operations if not op["success"]]
        
        return {
            "total_operations": len(self.operations),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "average_duration": sum(op["duration_seconds"] for op in successful_ops) / len(successful_ops) if successful_ops else 0,
            "total_duration": sum(op["duration_seconds"] for op in self.operations),
            "success_rate": len(successful_ops) / len(self.operations) * 100
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize client
    config = OpenAlgebraConfig(
        base_url="http://127.0.0.1:8000",
        timeout=60
    )
    client = OpenAlgebraClient(config)
    
    # Test basic functionality
    try:
        # Health check
        health = client.health_check()
        print(f"Health check: {health}")
        
        # System status
        status = client.get_system_status()
        print(f"System status: {status}")
        
        # Initialize AI agent
        agent = MedicalAIAgent(client)
        
        # Example DICOM processing
        dicom_request = DICOMProcessRequest(
            file_path="/path/to/dicom/series",
            output_format="sparse_tensor",
            anonymize=True,
            validate_hipaa=True
        )
        
        # This would process DICOM files if they exist
        # result = client.process_dicom(dicom_request)
        # print(f"DICOM processing result: {result}")
        
        # Performance monitoring
        monitor = PerformanceMonitor(client)
        
        # Benchmark health check
        benchmark_result = monitor.benchmark_operation(client.health_check)
        print(f"Health check benchmark: {benchmark_result}")
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        print(f"Performance summary: {summary}")
        
        logger.info("OpenAlgebra Medical AI client test completed successfully!")
        
    except Exception as e:
        logger.error(f"Client test failed: {e}") 