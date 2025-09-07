#!/usr/bin/env python3
"""
Upload script for forgeNN package to PyPI.
"""

import os
import subprocess
import sys
import getpass
from pathlib import Path

def run_command(cmd, cwd=None, input_data=None):
    """Run a command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True, input=input_data)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False, e.stderr

def check_dist_files():
    """Check if distribution files exist."""
    print("🔍 Checking distribution files...")
    
    if not os.path.exists('dist'):
        print("❌ dist/ directory not found. Run build_package.py first.")
        return False
    
    files = list(Path('dist').glob('*'))
    if not files:
        print("❌ No files found in dist/. Run build_package.py first.")
        return False
    
    print("📦 Found distribution files:")
    for file in files:
        size = os.path.getsize(file)
        print(f"   📄 {file.name} ({size:,} bytes)")
    
    return True

def check_credentials():
    """Check if PyPI credentials are configured."""
    print("🔍 Checking PyPI credentials...")
    
    # Check for .pypirc file
    pypirc_path = Path.home() / '.pypirc'
    if pypirc_path.exists():
        print("✅ Found .pypirc file")
        return True
    
    # Check for environment variables
    if 'TWINE_USERNAME' in os.environ and 'TWINE_PASSWORD' in os.environ:
        print("✅ Found credentials in environment variables")
        return True
    
    print("⚠️  No PyPI credentials found")
    print("   You can:")
    print("   1. Create ~/.pypirc file")
    print("   2. Set TWINE_USERNAME and TWINE_PASSWORD environment variables")
    print("   3. Enter credentials manually when prompted")
    
    return True

def upload_to_testpypi():
    """Upload package to TestPyPI."""
    print("🚀 Uploading to TestPyPI...")
    
    cmd = f"{sys.executable} -m twine upload --repository testpypi dist/*"
    success, output = run_command(cmd)
    
    if success:
        print("✅ Successfully uploaded to TestPyPI!")
        print("\n🧪 Test your package:")
        print("   pip install --index-url https://test.pypi.org/simple/ forgeNN")
        return True
    else:
        print("❌ Upload to TestPyPI failed")
        return False

def upload_to_pypi():
    """Upload package to PyPI."""
    print("🚀 Uploading to PyPI...")
    
    # Final confirmation
    response = input("\n⚠️  This will upload to production PyPI. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Upload cancelled.")
        return False
    
    cmd = f"{sys.executable} -m twine upload dist/*"
    success, output = run_command(cmd)
    
    if success:
        print("✅ Successfully uploaded to PyPI!")
        print("\n🎉 Your package is now available:")
        print("   pip install forgeNN")
        print("   https://pypi.org/project/forgeNN/")
        return True
    else:
        print("❌ Upload to PyPI failed")
        return False

def verify_upload():
    """Verify the uploaded package."""
    print("🔍 Verifying upload...")
    
    print("\n1. Check PyPI page: https://pypi.org/project/forgeNN/")
    print("2. Test installation in a fresh environment:")
    print("   python -m venv test_env")
    print("   test_env\\Scripts\\activate  # Windows")
    print("   # source test_env/bin/activate  # Linux/Mac")
    print("   pip install forgeNN")
    print("   python -c 'import forgeNN; print(forgeNN.__version__)'")

def main():
    """Main upload process."""
    print("="*60)
    print("📤 forgeNN Package Upload Script")
    print("="*60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check prerequisites
    if not check_dist_files():
        sys.exit(1)
    
    check_credentials()
    
    print("\n" + "="*60)
    print("UPLOAD OPTIONS")
    print("="*60)
    
    print("1. TestPyPI (recommended first)")
    print("2. PyPI (production)")
    print("3. Both (TestPyPI first, then PyPI)")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            upload_to_testpypi()
            break
        elif choice == '2':
            upload_to_pypi()
            break
        elif choice == '3':
            if upload_to_testpypi():
                print("\n" + "="*40)
                print("TestPyPI upload successful!")
                print("Testing before PyPI upload...")
                response = input("Continue with PyPI upload? (yes/no): ")
                if response.lower() == 'yes':
                    upload_to_pypi()
            break
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-4.")
    
    print("\n" + "="*60)
    print("📋 POST-UPLOAD CHECKLIST")
    print("="*60)
    verify_upload()
    
    print("\n✅ Upload process completed!")

if __name__ == "__main__":
    main()
