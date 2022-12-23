# -*- coding: utf-8 -*-
"""Setup configuration."""
import os

import numpy as np
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = ".pyx" if USE_CYTHON else ".c"
extensions = [Extension("pulse3D.compression_cy", [os.path.join("src", "pulse3D", "compression_cy") + ext])]
if USE_CYTHON:
    # cythonizing compression_cy.pyx with kwarg annotate=True will help when optimizing the code by enabling generation of the html annotation file
    extensions = cythonize(extensions, annotate=False)

setup(
    name="Pulse3D",
    version="0.29.0",
    description="Pulse3D Analysis Platform",
    url="https://github.com/CuriBio/Pulse3D",
    project_urls={"Documentation": "https://pulse3D.readthedocs.io/en/latest/"},
    author="Curi Bio",
    author_email="contact@curibio.com",
    license="MIT",
    package_dir={"": "src"},
    include_dirs=[np.get_include()],
    packages=find_packages("src"),
    install_requires=[
        "h5py>=3.7.0",
        "nptyping==1.4.4",  # Tanner (4/7/22): pinning for now, can upgrade to 2.0.0 once there is time to refactor
        "numpy>=1.23.4",
        "scipy>=1.9.3",
        "numba==0.56.4",
        "immutabledict>=1.2.0",
        "XlsxWriter>=1.3.8",
        "openpyxl>=3.0.7",
        "stdlib_utils>=0.4.4",
        "labware-domain-models>=0.3.1",
        "requests>=2.25.1",
        "semver>=2.13.0",
        "pandas>=1.3.4",
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
    ext_modules=extensions,
)
