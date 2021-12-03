# -*- coding: utf-8 -*-
"""Setup configuration."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="Pulse3D",
    version="0.18.1",
    description="CREATE A DESCRIPTION",
    url="https://github.com/CuriBio/Pulse3D",
    project_urls={"Documentation": "https://pulse3D.readthedocs.io/en/latest/"},
    author="Curi Bio",
    author_email="contact@curibio.com",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "h5py>=3.2.1",
        "nptyping>=1.4.1",
        "numpy>=1.20.1",
        "immutabledict>=1.2.0",
        "XlsxWriter>=1.3.8",
        "openpyxl>=3.0.7",
        "matplotlib>=3.4.1",
        "stdlib_utils>=0.4.4",
        "labware-domain-models>=0.3.1",
        "requests>=2.25.1",
        'importlib-metadata >= 3.7.3 ; python_version < "3.8"',
    ],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
