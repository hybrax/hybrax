from setuptools import setup, find_packages

setup(
    name="bpbench",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)
