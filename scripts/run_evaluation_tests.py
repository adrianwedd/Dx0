#!/usr/bin/env python3
"""Comprehensive test runner for Dx0 evaluation system tests.

This script provides a unified interface for running the complete evaluation
testing suite, including metrics validation, statistical analysis verification,
performance benchmarking, and integration testing.
"""

import argparse
import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile


class EvaluationTestRunner:
    """Comprehensive test runner for evaluation system validation."""
    
    def __init__(self, project_root: str = None):
        """Initialize test runner.
        
        Parameters
        ----------
        project_root : str, optional
            Path to project root directory.
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.test_results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, parallel: bool = False, verbose: bool = True, 
                     skip_slow: bool = False) -> Dict[str, Any]:
        """Run all evaluation tests.
        
        Parameters
        ----------
        parallel : bool
            Whether to run tests in parallel.
        verbose : bool
            Whether to use verbose output.
        skip_slow : bool
            Whether to skip slow tests.
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive test results.
        """
        self.start_time = datetime.now()
        
        print("🔬 Starting Dx0 Evaluation Testing Suite")
        print("=" * 50)
        
        # Define test categories
        test_categories = [
            ("metrics", "test_evaluation_metrics.py", "Diagnostic Accuracy & Performance Metrics"),
            ("statistics", "test_statistical_analysis.py", "Statistical Analysis & Significance Testing"),
            ("frameworks", "test_evaluation_frameworks.py", "Evaluation Frameworks & Benchmarking"),
            ("performance", "test_evaluation_performance.py", "Performance & Integration Testing"),
            ("data", "test_evaluation_data.py", "Test Data Generation & Validation")
        ]
        
        # Run each test category
        for category, test_file, description in test_categories:
            print(f"\n📊 Running {description}")
            print("-" * 40)
            
            result = self._run_test_category(
                category, test_file, parallel=parallel, verbose=verbose, skip_slow=skip_slow
            )
            self.test_results[category] = result
            
            # Print summary
            if result["success"]:
                print(f"✅ {description}: PASSED ({result['duration']:.1f}s)")
            else:
                print(f"❌ {description}: FAILED ({result['duration']:.1f}s)")
                if not verbose:
                    print(f"   Error: {result.get('error', 'Unknown error')}")
        
        self.end_time = datetime.now()
        
        # Generate comprehensive summary
        summary = self._generate_test_summary()
        
        # Print final results
        self._print_final_summary(summary)
        
        return summary
    
    def run_specific_tests(self, test_categories: List[str], **kwargs) -> Dict[str, Any]:
        """Run specific test categories.
        
        Parameters
        ----------
        test_categories : List[str]
            List of test categories to run.
        **kwargs
            Additional arguments passed to run_all_tests.
            
        Returns
        -------
        Dict[str, Any]
            Test results for specified categories.
        """
        # Map category names to test files
        category_map = {
            "metrics": "test_evaluation_metrics.py",
            "statistics": "test_statistical_analysis.py", 
            "frameworks": "test_evaluation_frameworks.py",
            "performance": "test_evaluation_performance.py",
            "data": "test_evaluation_data.py"
        }
        
        self.start_time = datetime.now()
        
        print(f"🔬 Running Specific Evaluation Tests: {', '.join(test_categories)}")
        print("=" * 50)
        
        for category in test_categories:
            if category not in category_map:
                print(f"❌ Unknown test category: {category}")
                continue
            
            test_file = category_map[category]
            print(f"\n📊 Running {category} tests")
            print("-" * 30)
            
            result = self._run_test_category(category, test_file, **kwargs)
            self.test_results[category] = result
            
            if result["success"]:
                print(f"✅ {category}: PASSED ({result['duration']:.1f}s)")
            else:
                print(f"❌ {category}: FAILED ({result['duration']:.1f}s)")
        
        self.end_time = datetime.now()
        return self._generate_test_summary()
    
    def run_performance_benchmark(self, scale: str = "medium") -> Dict[str, Any]:
        """Run performance benchmarking suite.
        
        Parameters
        ----------
        scale : str
            Scale of benchmark (small, medium, large).
            
        Returns
        -------
        Dict[str, Any]
            Performance benchmark results.
        """
        print(f"🏃 Running Performance Benchmark (Scale: {scale})")
        print("=" * 50)
        
        # Set environment variables for benchmark
        env = os.environ.copy()
        env["BENCHMARK_SCALE"] = scale
        env["PYTEST_MARKERS"] = "benchmark"
        
        start_time = time.time()
        
        try:
            # Run performance tests with benchmark marker
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/test_evaluation_performance.py",
                "-v", "-m", "benchmark",
                "--tb=short"
            ], 
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
            )
            
            duration = time.time() - start_time
            
            benchmark_result = {
                "success": result.returncode == 0,
                "duration": duration,
                "scale": scale,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
            if benchmark_result["success"]:
                print(f"✅ Performance Benchmark: PASSED ({duration:.1f}s)")
            else:
                print(f"❌ Performance Benchmark: FAILED ({duration:.1f}s)")
                print(f"   Error: {result.stderr}")
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            print("⏰ Performance Benchmark: TIMEOUT")
            return {
                "success": False,
                "duration": 600.0,
                "scale": scale,
                "error": "Benchmark timed out after 10 minutes"
            }
    
    def generate_test_report(self, output_file: str = None) -> str:
        """Generate comprehensive test report.
        
        Parameters
        ----------
        output_file : str, optional
            Path to output file. If None, uses default naming.
            
        Returns
        -------
        str
            Path to generated report file.
        """
        if not self.test_results:
            raise ValueError("No test results available. Run tests first.")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.project_root / f"evaluation_test_report_{timestamp}.json"
        
        # Generate comprehensive report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "test_runner_version": "1.0.0",
                "project_root": str(self.project_root),
                "total_duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None
            },
            "summary": self._generate_test_summary(),
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        # Write report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: {output_file}")
        return str(output_file)
    
    def _run_test_category(self, category: str, test_file: str, parallel: bool = False,
                          verbose: bool = True, skip_slow: bool = False) -> Dict[str, Any]:
        """Run a specific test category.
        
        Parameters
        ----------
        category : str
            Test category name.
        test_file : str
            Test file name.
        parallel : bool
            Whether to run in parallel.
        verbose : bool
            Whether to use verbose output.
        skip_slow : bool
            Whether to skip slow tests.
            
        Returns
        -------
        Dict[str, Any]
            Test execution results.
        """
        test_path = self.project_root / "tests" / test_file
        
        if not test_path.exists():
            return {
                "success": False,
                "duration": 0.0,
                "error": f"Test file not found: {test_path}",
                "output": ""
            }
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", str(test_path)]
        
        if verbose:
            cmd.append("-v")
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        if skip_slow:
            cmd.extend(["-m", "not slow"])
        
        cmd.extend(["--tb=short", "--disable-warnings"])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per category
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "duration": duration,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "command": " ".join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "duration": 300.0,
                "error": f"Test category {category} timed out after 5 minutes",
                "output": ""
            }
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate summary of test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(result["duration"] for result in self.test_results.values())
        
        return {
            "total_categories": total_tests,
            "passed_categories": passed_tests,
            "failed_categories": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_duration": total_duration,
            "overall_success": failed_tests == 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed tests
        failed_categories = [
            category for category, result in self.test_results.items()
            if not result["success"]
        ]
        
        if failed_categories:
            recommendations.append(
                f"Address failures in: {', '.join(failed_categories)}"
            )
        
        # Check for slow tests
        slow_categories = [
            category for category, result in self.test_results.items()
            if result["duration"] > 60.0  # More than 1 minute
        ]
        
        if slow_categories:
            recommendations.append(
                f"Consider optimizing slow test categories: {', '.join(slow_categories)}"
            )
        
        # Check overall performance
        total_duration = sum(result["duration"] for result in self.test_results.values())
        if total_duration > 300.0:  # More than 5 minutes total
            recommendations.append(
                "Consider running tests in parallel to reduce total execution time"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passed successfully! Consider adding more edge cases.")
        
        return recommendations
    
    def _print_final_summary(self, summary: Dict[str, Any]) -> None:
        """Print final test summary."""
        print("\n" + "=" * 50)
        print("🎯 EVALUATION TESTING SUMMARY")
        print("=" * 50)
        
        print(f"📊 Test Categories: {summary['passed_categories']}/{summary['total_categories']} passed")
        print(f"⏱️  Total Duration: {summary['total_duration']:.1f} seconds")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        
        if summary['overall_success']:
            print("\n🎉 ALL EVALUATION TESTS PASSED!")
        else:
            print(f"\n⚠️  {summary['failed_categories']} test categories failed")
        
        # Print recommendations
        if "recommendations" in summary:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 50)


def main():
    """Main entry point for evaluation test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Dx0 evaluation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/run_evaluation_tests.py --all
  
  # Run specific test categories
  python scripts/run_evaluation_tests.py --categories metrics statistics
  
  # Run performance benchmark
  python scripts/run_evaluation_tests.py --benchmark --scale large
  
  # Run tests in parallel with verbose output
  python scripts/run_evaluation_tests.py --all --parallel --verbose
  
  # Skip slow tests for quick validation
  python scripts/run_evaluation_tests.py --all --skip-slow
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--all", action="store_true",
                           help="Run all evaluation tests")
    test_group.add_argument("--categories", nargs="+", 
                           choices=["metrics", "statistics", "frameworks", "performance", "data"],
                           help="Run specific test categories")
    test_group.add_argument("--benchmark", action="store_true",
                           help="Run performance benchmark only")
    
    # Execution options
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Use verbose output")
    parser.add_argument("--skip-slow", action="store_true",
                       help="Skip slow tests")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium",
                       help="Scale for performance benchmark")
    
    # Output options
    parser.add_argument("--report", type=str,
                       help="Generate test report to specified file")
    parser.add_argument("--project-root", type=str,
                       help="Path to project root directory")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = EvaluationTestRunner(project_root=args.project_root)
    
    try:
        # Execute requested tests
        if args.all:
            results = runner.run_all_tests(
                parallel=args.parallel,
                verbose=args.verbose,
                skip_slow=args.skip_slow
            )
        elif args.categories:
            results = runner.run_specific_tests(
                test_categories=args.categories,
                parallel=args.parallel,
                verbose=args.verbose,
                skip_slow=args.skip_slow
            )
        elif args.benchmark:
            results = runner.run_performance_benchmark(scale=args.scale)
        
        # Generate report if requested
        if args.report:
            report_path = runner.generate_test_report(args.report)
            print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Exit with appropriate code
        if results.get("overall_success", False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⛔ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()