from setuptools import setup, find_packages

setup(
    name="terrain_generator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "matplotlib>=3.3.3",
        "open3d",
        "networkx",
    ],
    entry_points={
        "console_scripts": [
            "terrain-generator=terrain_generator.cli:main",
        ],
    },
    package_data={"terrain_generator": ["*.py"]},
    description="A Python library for generating terrain meshes",
    author="Takahiro Miki",
    author_email="takahiro.miki1992@gmail.com",
    url="https://github.com/mktk1117/terrain_generator",
)
