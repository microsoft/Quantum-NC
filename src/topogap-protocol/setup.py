#!/bin/env python
# -*- coding: utf-8 -*-
##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##

# IMPORTS #

import setuptools
import os
import distutils

# VERSION INFORMATION #
# Our build process sets the PYTHON_VERSION environment variable to a version
# string that is compatible with PEP 440, and so we inherit that version number
# here and propagate that to qsharp/version.py.
#
# To make sure that local builds still work without the same environment
# variables, we'll default to 0.0.0.1 as a development version.

version = os.environ.get("PYTHON_VERSION", "0.0.0.1")

with open("./topogap_protocol/version.py", "w") as f:
    f.write(
        f"""# Auto-generated file, do not edit.
##
# version.py: Specifies the version of the azure.quantum package.
##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under MICROSOFT QUANTUM NON-COMMERCIAL License.
##
__version__ = "{version}"
"""
    )

# DESCRIPTION #
# The long description metadata passed to setuptools is used to populate the
# PyPI page for this package. Thus, we'll generate the description by using the
# same README.md file that we use in the GitHub repo.

with open("./README.md", "r") as fh:
    long_description = fh.read()

# LIST OF REQUIREMENTS #
# Get list of requirements from requirements.txt
with open("./requirements.txt", "r") as fh:
    requirements = fh.readlines()

# SETUPTOOLS INVOCATION #
setuptools.setup(
    name="azure-quantum-topogap-protocol",
    version=version,
    author="Microsoft",
    description="Azure Quantum Topogap Protocol Utilities for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/Quantum-NC/tree/main/src/topogap-protocol",
    packages= ['topogap_protocol'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
