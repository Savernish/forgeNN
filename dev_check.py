#!/usr/bin/env python3
"""
Continuous Integration Script for forgeNN
=========================================

Quick test runner for development workflow.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run development checks."""
    print("🚀 forgeNN Development Checks")
    print("=" * 40)
    
    checks = [
        ("cd tests && python run_tests.py", "Unit Tests"),
        ("python -c \"import sys; sys.path.insert(0, '.'); from forgeNN.tensor import Tensor; print('Import check passed')\"", "Import Check"),
    ]
    
    passed = 0
    total = len(checks)
    
    for cmd, description in checks:
        if run_command(cmd, description):
            passed += 1
    
    print(f"\n{'=' * 40}")
    print(f"📊 SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Ready for development.")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
