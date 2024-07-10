from setuptools import setup, find_packages

setup(
    name="ASRhandler",
    packages=find_packages(),
    install_requires=[
        "pydantic==2.6.3",
        "pydantic-settings==2.2.1"
    ]
)
