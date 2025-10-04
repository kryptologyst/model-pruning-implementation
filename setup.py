"""
Setup script for Model Pruning Implementation project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="model-pruning-implementation",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="A comprehensive implementation of neural network pruning techniques with modern tools and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/model-pruning-implementation",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "web": [
            "streamlit>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "model-pruning=src.model_pruning:main",
            "pruning-ui=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    keywords="machine-learning neural-networks pruning pytorch optimization",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/model-pruning-implementation/issues",
        "Source": "https://github.com/yourusername/model-pruning-implementation",
        "Documentation": "https://github.com/yourusername/model-pruning-implementation#readme",
    },
)
