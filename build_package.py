#!/usr/bin/env python3
"""
Build script for forgeNN package distribution.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ['build', 'dist', 'forgeNN.egg-info']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"   Removed {dir_name}/")
            except Exception as e:
                print(f"   Warning: Could not remove {dir_name}/: {e}")
    
    # Clean __pycache__ directories
    try:
        for root, dirs, files in os.walk('.'):
            for dir_name in dirs[:]:
                if dir_name == '__pycache__':
                    try:
                        cache_path = os.path.join(root, dir_name)
                        shutil.rmtree(cache_path)
                        dirs.remove(dir_name)
                        print(f"   Removed {cache_path}")
                    except Exception as e:
                        print(f"   Warning: Could not remove {cache_path}: {e}")
    except Exception as e:
        print(f"   Warning: Error during cache cleanup: {e}")
    
    return True

def check_dependencies():
    """Check if required build dependencies are installed."""
    print("🔍 Checking build dependencies...")
    
    required = ['build', 'twine']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"   ❌ {package}")
    
    if missing:
        print(f"\n📦 Installing missing dependencies: {', '.join(missing)}")
        cmd = f"{sys.executable} -m pip install {' '.join(missing)}"
        if not run_command(cmd):
            print("❌ Failed to install dependencies")
            return False
    
    return True

def validate_package():
    """Validate package configuration."""
    print("🔍 Validating package configuration...")
    
    required_files = [
        'pyproject.toml',
        'README.md', 
        'LICENSE',
        'forgeNN/__init__.py'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"   ❌ Missing required file: {file_path}")
            return False
        print(f"   ✅ {file_path}")
    
    return True

def build_package():
    """Build the package."""
    print("🔨 Building package...")
    
    if not run_command(f"{sys.executable} -m build"):
        print("❌ Build failed")
        return False
    
    print("✅ Package built successfully")
    
    # List built files
    if os.path.exists('dist'):
        print("\n📦 Built packages:")
        for file in os.listdir('dist'):
            file_path = os.path.join('dist', file)
            size = os.path.getsize(file_path)
            print(f"   📄 {file} ({size:,} bytes)")
    
    return True

def check_package():
    """Check the built package."""
    print("🔍 Checking package with twine...")
    
    if not run_command(f"{sys.executable} -m twine check dist/*"):
        print("❌ Package check failed")
        return False
    
    print("✅ Package check passed")
    return True

def main():
    """Main build process."""
    print("="*60)
    print("🚀 forgeNN Package Build Script")
    print("="*60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    steps = [
        ("Checking dependencies", check_dependencies),
        ("Validating package", validate_package), 
        ("Cleaning build artifacts", clean_build),
        ("Building package", build_package),
        ("Checking package", check_package),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"\n❌ Failed at step: {step_name}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ BUILD SUCCESSFUL!")
    print("="*60)
    
    print("\n📦 Your package is ready for upload!")
    print("\nNext steps:")
    print("1. Test upload to TestPyPI:")
    print("   python -m twine upload --repository testpypi dist/*")
    print("\n2. Upload to PyPI:")
    print("   python -m twine upload dist/*")
    print("\n3. Install from PyPI:")
    print("   pip install forgeNN")

if __name__ == "__main__":
    main()
