#!/usr/bin/env python3
"""
Setup script for Loop Singular Bit
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Loop Singular Bit - Extreme Model Compression through Outlier-Preserving 1-Bit Quantization"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "safetensors>=0.3.0",
            "numpy>=1.24.0",
            "psutil>=5.9.0",
            "huggingface-hub>=0.15.0",
            "tqdm>=4.65.0",
            "requests>=2.28.0"
        ]

setup(
    name="loop-singular-bit",
    version="1.0.0",
    author="Bommareddy Bharath Reddy",
    author_email="contact@loop.org",
    description="Extreme Model Compression through Outlier-Preserving 1-Bit Quantization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/rockstaaa/loop-singular-bit",
    packages=find_packages(),
    py_modules=["loop_singular_bit"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "safetensors>=0.3.0",
            "numpy>=1.24.0",
            "psutil>=5.9.0",
            "huggingface-hub>=0.15.0",
            "tqdm>=4.65.0",
            "requests>=2.28.0",
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "loop-compress=loop_singular_bit:load_compressed_model",
            "loop-list=loop_singular_bit:list_models",
            "loop-info=loop_singular_bit:get_system_info",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/rockstaaa/loop-singular-bit/issues",
        "Source": "https://github.com/rockstaaa/loop-singular-bit",
        "Documentation": "https://github.com/rockstaaa/loop-singular-bit/blob/main/README.md",
    },
    keywords=[
        "machine learning",
        "model compression", 
        "quantization",
        "1-bit",
        "outlier preserving",
        "memory optimization",
        "inference acceleration",
        "transformer compression",
        "llm compression",
        "ai optimization"
    ],
    package_data={
        "loop_singular_bit": [
            "models/compressed/*.json",
            "compression/*.py",
            "docs/*.md",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    zip_safe=False,
)
