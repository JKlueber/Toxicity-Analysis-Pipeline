from setuptools import setup, find_packages

setup(
    name="toxicity_classifier",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "toxicity=src.toxic_bert:main",
        ],
    },
)