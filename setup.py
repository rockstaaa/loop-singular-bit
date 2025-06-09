#!/usr/bin/env python3
"""
Setup script for Loop Singular Bit
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="loop-singular-bit",
    version="1.0.0",
    author="Bommareddy Bharath Reddy",
    author_email="contact@loop.org",
    description="Extreme Model Compression through Outlier-Preserving 1-Bit Quantization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rockstaaa/loop-singular-bit",
    packages=find_packages(),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "safetensors>=0.3.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
        "huggingface-hub>=0.15.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "loop-compress=loop_singular_bit.cli:compress_model",
            "loop-decompress=loop_singular_bit.cli:decompress_model",
            "loop-benchmark=loop_singular_bit.cli:benchmark_model",
        ],
    },
    include_package_data=True,
)
