from setuptools import setup, find_packages
import os

setup(
    name='cutskycov',
    version='0.0.1',
    author='Nick Hand',
    packages=find_packages(),
    description=("Python code to compute analytic cutsky covariances"),
    install_requires=['numpy', 'scipy'],
)
