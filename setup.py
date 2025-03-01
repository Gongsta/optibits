from setuptools import setup, find_packages

setup(
    name="optibits",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "numpy",
        "matplotlib",
        "bitsandbytes",
    ],
)
