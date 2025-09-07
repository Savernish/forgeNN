#!/usr/bin/env python3
"""
forgeNN Setup Configuration
===========================

Installation script for forgeNN neural network framework.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read README.md for the long description."""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    here = os.path.abspath(os.path.dirname(__file__))
    requirements_path = os.path.join(here, 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Only include core dependencies, not dev dependencies
                    if any(pkg in line for pkg in ['numpy', 'scikit-learn']):
                        requirements.append(line)
    return requirements

# Package metadata
setup(
    name="forgeNN",
    version="1.0.0",
    author="Savernish",
    author_email="",  # Add your email if desired
    description="A From Scratch Neural Network Framework with Educational Purposes",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Savernish/forgeNN",
    project_urls={
        "Bug Reports": "https://github.com/Savernish/forgeNN/issues",
        "Source": "https://github.com/Savernish/forgeNN",
        "Documentation": "https://github.com/Savernish/forgeNN/blob/main/README.md",
    },
    
    # Package discovery
    packages=find_packages(),
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'examples': [
            'matplotlib>=3.5.0',
            'jupyter>=1.0.0',
        ],
        'all': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'matplotlib>=3.5.0',
            'jupyter>=1.0.0',
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "neural-networks",
        "machine-learning",
        "deep-learning",
        "education",
        "automatic-differentiation",
        "numpy",
        "vectorized",
        "from-scratch",
        "ai",
        "artificial-intelligence",
    ],
    
    # Include additional files
    include_package_data=True,
    package_data={
        'forgeNN': ['*.md', '*.txt'],
    },
    
    # Entry points (if you want to provide command-line tools)
    entry_points={
        'console_scripts': [
            # Add command-line scripts here if needed
            # 'forgenn-demo=forgeNN.cli:main',
        ],
    },
    
    # Zip safety
    zip_safe=False,
)