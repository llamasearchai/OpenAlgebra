#!/usr/bin/env python3
"""
Generate JUnit XML reports from test results
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from datetime import datetime

def generate_junit_report(input_dir: str, output_file: str):
    """Generate JUnit XML report from test results"""
    
    input_path = Path(input_dir)
    test_suites = ET.Element("testsuites")
    
    # Process each test result file
    for log_file in input_path.glob("*.log"):
        test_suite = process_log_file(log_file)
        if test_suite is not None:
            test_suites.append(test_suite)
    
    # Write XML file
    tree = ET.ElementTree(test_suites)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
    print(f"JUnit report generated: {output_file}")

def process_log_file(log_file: Path) -> ET.Element:
    """Process individual log file and create test suite"""
    
    test_name = log_file.stem
    test_suite = ET.Element("testsuite")
    test_suite.set("name", test_name)
    test_suite.set("timestamp", datetime.now().isoformat())
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract test results
    tests = 0
    failures = 0
    errors = 0
    time = 0.0
    
    # Parse success rate
    success_match = re.search(r'Overall Success Rate: ([\d.]+)%', content)
    if success_match:
        success_rate = float(success_match.group(1)) / 100
        
        # Create test case for success rate
        test_case = ET.SubElement(test_suite, "testcase")
        test_case.set("classname", f"{test_name}.SuccessRate")
        test_case.set("name", "success_rate_check")
        test_case.set("time", "1.0")
        
        if success_rate < 0.95:
            failure = ET.SubElement(test_case, "failure")
            failure.set("message", f"Success rate too low: {success_rate:.1%}")
            failure.text = f"Expected >= 95%, got {success_rate:.1%}"
            failures += 1
        
        tests += 1
    
    # Parse compliance rate
    compliance_match = re.search(r'HIPAA Compliance Rate: ([\d.]+)%', content)
    if compliance_match:
        compliance_rate = float(compliance_match.group(1)) / 100
        
        test_case = ET.SubElement(test_suite, "testcase")
        test_case.set("classname", f"{test_name}.Compliance")
        test_case.set("name", "hipaa_compliance_check")
        test_case.set("time", "1.0")
        
        if compliance_rate < 0.99:
            failure = ET.SubElement(test_case, "failure")
            failure.set("message", f"HIPAA compliance too low: {compliance_rate:.1%}")
            failure.text = f"Expected >= 99%, got {compliance_rate:.1%}"
            failures += 1
        
        tests += 1
    
    # Parse latency metrics
    latency_match = re.search(r'Average Response Time: ([\d.]+)ms', content)
    if latency_match:
        avg_latency = float(latency_match.group(1))
        
        test_case = ET.SubElement(test_suite, "testcase")
        test_case.set("classname", f"{test_name}.Performance")
        test_case.set("name", "latency_check")
        test_case.set("time", str(avg_latency / 1000))
        
        if avg_latency > 1000:
            failure = ET.SubElement(test_case, "failure")
            failure.set("message", f"Average latency too high: {avg_latency}ms")
            failure.text = f"Expected <= 1000ms, got {avg_latency}ms"
            failures += 1
        
        tests += 1
        time += avg_latency / 1000
    
    # Set suite attributes
    test_suite.set("tests", str(tests))
    test_suite.set("failures", str(failures))
    test_suite.set("errors", str(errors))
    test_suite.set("time", str(time))
    
    return test_suite if tests > 0 else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JUnit XML report from test results")
    parser.add_argument("--input-dir", required=True, help="Directory containing test result files")
    parser.add_argument("--output-file", required=True, help="Output JUnit XML file")
    
    args = parser.parse_args()
    generate_junit_report(args.input_dir, args.output_file) 