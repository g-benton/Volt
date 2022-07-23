from setuptools import setup
import os
import sys

setup(
    name='voltron',
    version='alpha',
    description=('Voltron Repo'),
    author='Greg Benton',
    author_email='greg.w.benton@gmail.com',
    url='https://github.com/g-benton/voltron',
    license='Apache-2.0',
    packages=['voltron'],
    install_requires=[
    'matplotlib>=3.0.3',
    'setuptools>=41.0.0',
    'torch>=1.11.0',
    'numpy>=1.16.2',
    'gpytorch>=1.0.1',
    ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 0',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7'],
)