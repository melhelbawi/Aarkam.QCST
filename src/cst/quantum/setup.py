from setuptools import setup, find_packages

setup(
    name="qcst",
    version="0.1.0",
    description="Quantum-Enhanced Contextual Spectrum Tokenization",
    author="Mohamed Elhelbawi",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.0.0",
        "pennylane>=0.30.0",
        "transformers>=4.30.0",
        "numpy",
        "loguru"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
