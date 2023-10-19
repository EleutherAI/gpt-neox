import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mup",
    version="1.0.0",
    author="Edward J Hu, Greg Yang",
    author_email="edwardjhu@edwardjhu.com, gregyang@microsoft.com",
    description="Maximal Update Parametrization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/mup",
    download_url="https://github.com/microsoft/mup/archive/refs/tags/v1.0.0.tar.gz",
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'tqdm',
        'pyyaml'
      ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)