from setuptools import setup

with open("version.txt") as f:
    version = f.read().strip()

setup(
    version=version
)
