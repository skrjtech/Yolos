from setuptools import setup, find_packages

setup(
    name="yolos",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "yolos=src.__main__:main"
        ]
    }
)