# `CTApy`

Python package for the "Conditional Topic Allocation" (CTA): a text-analysis method that identifies topics that correlate with numerical outcomes.


* Corresponding research paper: [Conditional Topic Allocations for Open-Ended Survey Responses (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4190308).


## How does CTA work?


CTA finds topics by conditioning on observables. For example, do Republicans write differently about politics than Democrats?
It consists of three steps:

<br>
1. Predict the outcome variable with text.

* Uses DistilBERT to predict outcome.
 
 <br>
2. Select words with high predictive power (positive or negative).

* Calculates SHAP values for each word and select words with a statistically significant SHAP value.

<br>
3. Group words by semantic similarity.

* Returns topics with either positive or negative correlation with the outcome.

<br>
CTA supports all languages.

## Installation

CTApy requires Python 3.9 and pip.  
It is highly recommended to use a virtual environment (or conda environment) for the installation.

```bash
# upgrade pip, wheel and setuptools
python -m pip install -U pip wheel setuptools

# install the package
python -m pip install -U CTApy
```

If you want to use Jupyter, make sure you have it installed in the current environment.

## Quickstart 

Please see the hands-on tutorials, which replicate the research paper: [https://github.com/twekhof/CTA/tree/main/tutorials](https://github.com/twekhof/CTA/tree/main/tutorials).


## Author

`CTApy` was developed by

[Tobias Wekhof](https://tobiaswekhof.com), ETH Zurich


## Disclaimer

This Python package is a research tool currently under development. The authors take no responsibility for the accuracy or reliability of the results produced by it.