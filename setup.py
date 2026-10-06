from setuptools import find_packages, setup

setup(
    name="bottled",
    version="0.1.0",
    description="BOTTLED",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
)
