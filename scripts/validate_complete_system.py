#!/usr/bin/env python3
"""
Complete System Validation Script for OpenAlgebra Medical AI
Validates all components, tests, configurations, and deployment readiness.
"""

import os
import sys
import subprocess
import json
import yaml
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SystemValidator:
    """Comprehensive system validation for OpenAlgebra Medical AI."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.validation_results = {
            "project_structure": False,
            "dependencies": False,
            "configuration": False,
            "tests": False,
            "docker": False,
            "api": False,
            "workflows": False,
            "documentation": False
        }
        self.errors = []
        self.warnings = []
    
    def validate_project_structure(self) -> bool:
        """Validate project directory structure."""
        print("Validating project structure...")
        
        required_dirs = [
            "src", "tests", "scripts", "config", "docs",
            ".github/workflows", "examples", "python"
        ]
        
        required_files = [
            "README.md", "requirements.txt", "Dockerfile",
            "docker-compose.yml", "Cargo.toml", "CMakeLists.txt"
        ]
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                self.errors.append(f"Missing directory: {dir_path}")
                return False
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.errors.append(f"Missing file: {file_path}")
                return False
        
        print("✓ Project structure validation passed")
        return True
    
    def validate_dependencies(self) -> bool:
        """Validate Python dependencies."""
        print("Validating Python dependencies...")
        
        try:
            # Check if requirements.txt exists and is readable
            req_file = self.project_root / "requirements.txt"
            with open(req_file, 'r') as f:
                requirements = f.read().strip().split('\n')
            
            # Check critical dependencies
            critical_deps = [
                'fastapi', 'uvicorn', 'numpy', 'pytest',
                'pydantic', 'python-multipart'
            ]
            
            found_deps = []
            for req in requirements:
                if req.strip() and not req.strip().startswith('#'):
                    package_name = req.split('==')[0].split('>=')[0].split('<=')[0]
                    found_deps.append(package_name.strip())
            
            missing_deps = []
            for dep in critical_deps:
                if dep not in found_deps:
                    missing_deps.append(dep)
            
            if missing_deps:
                self.errors.append(f"Missing critical dependencies: {missing_deps}")
                return False
            
            print("✓ Dependencies validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating dependencies: {e}")
            return False
    
    def validate_configuration(self) -> bool:
        """Validate configuration files."""
        print("Validating configuration files...")
        
        try:
            # Check pytest configuration
            pytest_config = self.project_root / "pytest.ini"
            if pytest_config.exists():
                with open(pytest_config, 'r') as f:
                    config_content = f.read()
                    if '[tool:pytest]' not in config_content and '[pytest]' not in config_content:
                        self.warnings.append("pytest.ini may be improperly formatted")
            
            # Check Docker configuration
            docker_file = self.project_root / "Dockerfile"
            if docker_file.exists():
                with open(docker_file, 'r') as f:
                    dockerfile_content = f.read()
                    if 'FROM' not in dockerfile_content:
                        self.errors.append("Dockerfile missing FROM instruction")
                        return False
            
            # Check docker-compose configuration
            compose_file = self.project_root / "docker-compose.yml"
            if compose_file.exists():
                with open(compose_file, 'r') as f:
                    try:
                        compose_config = yaml.safe_load(f)
                        if 'services' not in compose_config:
                            self.errors.append("docker-compose.yml missing services section")
                            return False
                    except yaml.YAMLError as e:
                        self.errors.append(f"Invalid YAML in docker-compose.yml: {e}")
                        return False
            
            print("✓ Configuration validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating configuration: {e}")
            return False
    
    def validate_tests(self) -> bool:
        """Validate test suite."""
        print("Validating test suite...")
        
        try:
            # Run pytest to ensure tests pass
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_medical_ai.py", 
                "-v", "--tb=short"
            ], 
            cwd=self.project_root,
            capture_output=True, 
            text=True
            )
            
            if result.returncode != 0:
                self.errors.append(f"Tests failed:\n{result.stdout}\n{result.stderr}")
                return False
            
            # Check test coverage
            test_files = list((self.project_root / "tests").glob("test_*.py"))
            if len(test_files) == 0:
                self.errors.append("No test files found")
                return False
            
            print("✓ Test suite validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Error running tests: {e}")
            return False
    
    def validate_docker(self) -> bool:
        """Validate Docker configuration."""
        print("Validating Docker configuration...")
        
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                self.warnings.append("Docker not available for validation")
                return True  # Don't fail if Docker isn't available
            
            # Validate docker-compose syntax
            result = subprocess.run([
                "docker-compose", "config"
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True
            )
            
            if result.returncode != 0:
                self.errors.append(f"Docker compose configuration invalid: {result.stderr}")
                return False
            
            print("✓ Docker validation passed")
            return True
            
        except FileNotFoundError:
            self.warnings.append("Docker not installed - skipping Docker validation")
            return True
        except Exception as e:
            self.errors.append(f"Error validating Docker: {e}")
            return False
    
    def validate_api_structure(self) -> bool:
        """Validate API structure and components."""
        print("Validating API structure...")
        
        try:
            # Check for API files
            api_files = [
                "src/api/medical_ai_service.py",
                "scripts/health_check.py"
            ]
            
            for api_file in api_files:
                full_path = self.project_root / api_file
                if not full_path.exists():
                    self.errors.append(f"Missing API file: {api_file}")
                    return False
                
                # Basic syntax check
                with open(full_path, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        self.errors.append(f"Empty API file: {api_file}")
                        return False
            
            print("✓ API structure validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating API structure: {e}")
            return False
    
    def validate_workflows(self) -> bool:
        """Validate GitHub Actions workflows."""
        print("Validating GitHub Actions workflows...")
        
        try:
            workflow_dir = self.project_root / ".github" / "workflows"
            workflow_files = list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))
            
            if len(workflow_files) == 0:
                self.errors.append("No GitHub Actions workflow files found")
                return False
            
            for workflow_file in workflow_files:
                with open(workflow_file, 'r') as f:
                    try:
                        workflow_config = yaml.safe_load(f)
                        
                        # Check required sections
                        required_sections = ['name', 'jobs']  # 'on' can also be 'true' for some workflows
                        for section in required_sections:
                            if section not in workflow_config:
                                self.errors.append(f"Workflow {workflow_file.name} missing {section} section")
                                return False
                        
                        # Check for 'on' or trigger configuration (can be 'on' key or True boolean)
                        has_trigger = (
                            'on' in workflow_config or
                            True in workflow_config or  # YAML parsing quirk
                            any(key in workflow_config for key in ['push', 'pull_request', 'schedule', 'workflow_dispatch'])
                        )
                        if not has_trigger:
                            self.errors.append(f"Workflow {workflow_file.name} missing trigger configuration")
                            return False
                                
                    except yaml.YAMLError as e:
                        self.errors.append(f"Invalid YAML in {workflow_file.name}: {e}")
                        return False
            
            print("✓ Workflows validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating workflows: {e}")
            return False
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        print("Validating documentation...")
        
        try:
            readme_file = self.project_root / "README.md"
            with open(readme_file, 'r') as f:
                readme_content = f.read()
            
            # Check for essential sections
            required_sections = [
                "# OpenAlgebra",
                "## Quick Start",
                "## Installation",
                "## License"
            ]
            
            for section in required_sections:
                if section not in readme_content:
                    self.errors.append(f"README.md missing section: {section}")
                    return False
            
            # Check for no emojis (as requested)
            common_emojis = ["🚀", "📊", "🧬", "⚡", "🔒", "🏥", "🎓", "👥"]
            found_emojis = []
            for emoji in common_emojis:
                if emoji in readme_content:
                    found_emojis.append(emoji)
            
            if found_emojis:
                self.errors.append(f"Found emojis in README.md: {found_emojis}")
                return False
            
            print("✓ Documentation validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Error validating documentation: {e}")
            return False
    
    def run_full_validation(self) -> Dict[str, bool]:
        """Run complete system validation."""
        print("=" * 60)
        print("OpenAlgebra Medical AI - Complete System Validation")
        print("=" * 60)
        
        # Run all validations
        validations = [
            ("project_structure", self.validate_project_structure),
            ("dependencies", self.validate_dependencies),
            ("configuration", self.validate_configuration),
            ("tests", self.validate_tests),
            ("docker", self.validate_docker),
            ("api", self.validate_api_structure),
            ("workflows", self.validate_workflows),
            ("documentation", self.validate_documentation)
        ]
        
        for name, validator in validations:
            try:
                self.validation_results[name] = validator()
            except Exception as e:
                self.errors.append(f"Validation {name} failed with exception: {e}")
                self.validation_results[name] = False
        
        # Print results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        all_passed = True
        for name, result in self.validation_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{name.replace('_', ' ').title():.<30} {status}")
            if not result:
                all_passed = False
        
        # Print errors and warnings
        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL VALIDATIONS PASSED - SYSTEM READY FOR PUBLICATION!")
        else:
            print("SOME VALIDATIONS FAILED - PLEASE FIX ERRORS BEFORE PUBLICATION")
        print("=" * 60)
        
        return self.validation_results


def main():
    """Main validation entry point."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    validator = SystemValidator(project_root)
    results = validator.run_full_validation()
    
    # Exit with error code if any validation failed
    if not all(results.values()):
        sys.exit(1)
    
    print("\nValidation complete. System is ready for publication!")
    sys.exit(0)


if __name__ == "__main__":
    main() 