# setup.py
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="ppo_from_scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
)
