import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TurkishPreprocessor",
    version="0.0.1",
    author="Yusuf ANI",
    author_email="yusufani8@gmail.com",
    description="A Turkish Preprocessor for NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yusufani/TurkishPreprocessor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)