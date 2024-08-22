import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="CTApy", 
    version="0.1.4",
    author="Tobias Wekhof",
    author_email="tobiaswekhof@gmail.com",
    description="Python package for the Conditional Topic Allocation (CTA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twekhof/CTA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    python_requires='>=3.9',
    install_requires=[
        'gensim>=4.3.2',
        'nltk>=3.8.1',
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'scipy>=1.11.1',
        'shap>=0.44.0',
        'spacy>=3.7.2',
        'torch>=1.2.1',
        'tqdm>=4.65.0',
        'transformers>=4.32.1'
    ]
    )