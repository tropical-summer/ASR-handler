from setuptools import setup, find_packages

setup(
    name="asrdiarization-handler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic==2.6.3",
        "pydantic-settings==2.2.1"
    ]
)