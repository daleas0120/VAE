#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='VAE',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "opencv-python",
        "pandas",
        "scikit-image",
        "scipy",
        "sklearn",
        "tabulate",
        "tensorflow",
        "tensorflow-gpu",
        "tqdm",
    ],
)