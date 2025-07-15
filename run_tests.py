#!/usr/bin/env python3
"""
Comprehensive test runner for MCP Yggdrasil
Executes the complete test suite with coverage reporting and performance metrics
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json


class TestRunner:
    """Comprehensive test runner with coverage and performance reporting."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.test_results = {}
        self.coverage_threshold = 50  # Minimum coverage percentage
        
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests with coverage."""
        print("🧪 Running unit tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--cov=agents",
            "--cov=cache",
            "--cov=api",
            "--cov=streamlit_workspace",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            f"--cov-fail-under={self.coverage_threshold}",
            "--markers",
            "-m", "unit"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        # Parse coverage from JSON file
        coverage_data = self._parse_coverage_json()
        
        return {
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'coverage': coverage_data,
            'return_code': result.returncode
        }
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "integration"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    
    def run_cache_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run cache system tests."""
        print("🗄️ Running cache system tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_cache_system.py",
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        print("⚡ Running performance tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "slow or performance",
            "--timeout=300"  # 5 minute timeout for performance tests
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    
    def run_specific_tests(self, test_pattern: str, verbose: bool = False) -> Dict[str, Any]:
        """Run specific tests matching a pattern."""
        print(f"🎯 Running specific tests: {test_pattern}")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-k", test_pattern,
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    
    def run_linting(self, verbose: bool = False) -> Dict[str, Any]:
        """Run linting checks."""
        print("🔍 Running linting checks...")
        
        linting_results = {}
        
        # Run flake8
        flake8_result = self._run_flake8()
        linting_results['flake8'] = flake8_result
        
        # Run black
        black_result = self._run_black()
        linting_results['black'] = black_result
        
        # Run mypy
        mypy_result = self._run_mypy()
        linting_results['mypy'] = mypy_result
        
        # Overall success
        overall_success = all(result['success'] for result in linting_results.values())
        
        return {
            'success': overall_success,
            'results': linting_results
        }
    
    def _run_flake8(self) -> Dict[str, Any]:
        """Run flake8 linting."""
        cmd = [sys.executable, "-m", "flake8", "agents", "cache", "api", "streamlit_workspace", "tests"]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    
    def _run_black(self) -> Dict[str, Any]:
        """Run black code formatting check."""
        cmd = [sys.executable, "-m", "black", "--check", "--diff", "agents", "cache", "api", "streamlit_workspace", "tests"]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    
    def _run_mypy(self) -> Dict[str, Any]:
        """Run mypy type checking."""
        cmd = [sys.executable, "-m", "mypy", "agents", "cache", "api"]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
    
    def _parse_coverage_json(self) -> Dict[str, Any]:
        """Parse coverage data from JSON file."""
        coverage_file = self.project_root / "coverage.json"
        
        if not coverage_file.exists():
            return {'total_coverage': 0, 'file_coverage': {}}
        
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            file_coverage = {}
            
            for filename, file_data in coverage_data.get('files', {}).items():
                file_coverage[filename] = file_data.get('summary', {}).get('percent_covered', 0)
            
            return {
                'total_coverage': total_coverage,
                'file_coverage': file_coverage,
                'meets_threshold': total_coverage >= self.coverage_threshold
            }
        
        except Exception as e:
            print(f"Error parsing coverage data: {e}")
            return {'total_coverage': 0, 'file_coverage': {}, 'error': str(e)}
    
    def run_comprehensive_suite(self, verbose: bool = False, skip_slow: bool = False) -> Dict[str, Any]:
        """Run the comprehensive test suite."""
        print("🚀 Running comprehensive test suite...")
        print("=" * 80)
        
        start_time = time.time()
        results = {}
        
        # Run linting first
        results['linting'] = self.run_linting(verbose)
        
        # Run unit tests
        results['unit_tests'] = self.run_unit_tests(verbose)
        
        # Run cache tests
        results['cache_tests'] = self.run_cache_tests(verbose)
        
        # Run integration tests
        results['integration_tests'] = self.run_integration_tests(verbose)
        
        # Run performance tests (if not skipped)
        if not skip_slow:
            results['performance_tests'] = self.run_performance_tests(verbose)
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(results, total_time)
        results['summary'] = summary
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate test summary."""
        summary = {
            'total_execution_time': total_time,
            'test_categories': {},
            'overall_success': True,
            'coverage_info': {}
        }
        
        for category, result in results.items():
            if category == 'summary':
                continue
                
            if isinstance(result, dict) and 'success' in result:
                summary['test_categories'][category] = {
                    'success': result['success'],
                    'execution_time': result.get('execution_time', 0)
                }
                
                if not result['success']:
                    summary['overall_success'] = False
                
                # Extract coverage info
                if 'coverage' in result:
                    summary['coverage_info'] = result['coverage']
        
        return summary
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        summary = results.get('summary', {})
        
        # Overall status
        status = "✅ PASSED" if summary.get('overall_success', False) else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Total Time: {summary.get('total_execution_time', 0):.2f}s")
        
        # Test categories
        print("\nTest Categories:")
        for category, info in summary.get('test_categories', {}).items():
            status = "✅" if info['success'] else "❌"
            time_str = f"{info['execution_time']:.2f}s"
            print(f"  {status} {category.replace('_', ' ').title()}: {time_str}")
        
        # Coverage info
        coverage_info = summary.get('coverage_info', {})
        if coverage_info:
            coverage_pct = coverage_info.get('total_coverage', 0)
            meets_threshold = coverage_info.get('meets_threshold', False)
            threshold_status = "✅" if meets_threshold else "❌"
            print(f"\nCoverage: {threshold_status} {coverage_pct:.1f}% (threshold: {self.coverage_threshold}%)")
        
        print("\n" + "=" * 80)
    
    def save_results(self, results: Dict[str, Any], filename: str = "test_results.json"):
        """Save test results to file."""
        output_file = self.project_root / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Test results saved to: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MCP Yggdrasil Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--cache", action="store_true", help="Run only cache tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--lint", action="store_true", help="Run only linting checks")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--test-pattern", "-k", help="Run tests matching pattern")
    parser.add_argument("--coverage-threshold", type=int, default=50, help="Coverage threshold percentage")
    parser.add_argument("--output", "-o", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    runner.coverage_threshold = args.coverage_threshold
    
    # Run specific test categories
    if args.unit:
        results = {'unit_tests': runner.run_unit_tests(args.verbose)}
    elif args.integration:
        results = {'integration_tests': runner.run_integration_tests(args.verbose)}
    elif args.cache:
        results = {'cache_tests': runner.run_cache_tests(args.verbose)}
    elif args.performance:
        results = {'performance_tests': runner.run_performance_tests(args.verbose)}
    elif args.lint:
        results = {'linting': runner.run_linting(args.verbose)}
    elif args.test_pattern:
        results = {'specific_tests': runner.run_specific_tests(args.test_pattern, args.verbose)}
    else:
        # Run comprehensive suite
        results = runner.run_comprehensive_suite(args.verbose, args.skip_slow)
    
    # Print summary
    runner.print_summary(results)
    
    # Save results if requested
    if args.output:
        runner.save_results(results, args.output)
    
    # Exit with appropriate code
    summary = results.get('summary', {})
    overall_success = summary.get('overall_success', False)
    
    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()