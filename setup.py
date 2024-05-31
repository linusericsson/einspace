#!/usr/bin/env python

from setuptools import find_packages, setup

print(f"Installing {find_packages()}")
setup(
    name="einspace",
    version="0.0.2",
    description="A space for architecture search in NNs that starts from low level primitives and assumes very minimal structure",
    author="xxxyyy",
    author_email="xxx@yyy",
    packages=find_packages(),
)
