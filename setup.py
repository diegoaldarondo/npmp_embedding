"""Setup file for dannce."""
from setuptools import setup, find_packages

setup(
    name="dispatch_embed",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "h5py",
        "pyyaml",
    ],
)
