#!/usr/bin/env python3
"""
Test runner script for MCP Server.
"""

import subprocess
import sys

def run_tests():
    print("🧪 Running MCP Server tests...")
    
    # Run different test suites
    test_commands = [
        ["pytest", "tests/unit/", "-v"],
        ["pytest", "tests/integration/", "-v"],
        ["pytest", "tests/performance/", "-v"],
        ["pytest", "--cov=agents", "--cov-report=html"]
    ]
    
    for cmd in test_commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Test failed: {' '.join(cmd)}")
            sys.exit(1)
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    run_tests()
