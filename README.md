[![Build Status](https://travis-ci.org/lacava/few.svg?branch=master)](https://travis-ci.org/lacava/few)
[![Code Health](https://landscape.io/github/lacava/few/master/landscape.svg?style=flat)](https://landscape.io/github/lacava/few/master)
[![Coverage Status](https://coveralls.io/repos/github/lacava/few/badge.svg?branch=master)](https://coveralls.io/github/lacava/few?branch=master)
[![DOI](https://zenodo.org/badge/65411376.svg)](https://zenodo.org/badge/latestdoi/65411376)

# Few

**Few** is a **Feature Engineering Wrapper** for scikit-learn. Few looks for a set of feature transformations that work best with a specified machine learning algorithm in order to improve model estimation and prediction. In doing so, Few is able to provide the user with a set of concise, engineered features that describe their data.

Few uses genetic programming to generate, search and update engineered features. It incorporates feedback from the ML process to select important features, while also scoring them internally. 


## Install

You can use pip to install FEW from [PyPi](https://pypi.python.org/pypi/FEW) as: 

```pip install few```

or you can clone the git repo and add it to your Python path. Then from the repo, run

```python setup.py install``` 

### Mac users 

Some Mac users have reported issues when installing with old versions of gcc (like gcc-4.2) because the random.h library is not included (basically [this issue](https://stackoverflow.com/questions/5967065/python-distutils-not-using-correct-version-of-gcc)). I recommend installing gcc-4.8 or greater for use with Few. After updating the compiler, you can reinstall with 

```python
CC=gcc-4.8 python setupy.py install
```

## Usage

Few uses the same nomenclature as [sklearn](http://scikit-learn.org/) supervised learning modules. Here is a simple example script:

```python
# import few
from few import FEW
# initialize
learner = FEW(generations=100, population_size=25, ml = LassoLarsCV())
# fit model
learner.fit(X,y)
# generate prediction
y_pred = learner.predict(X_unseen)
# get feature transformation
Phi = learner.transform(X_unseen)
```

You can also call Few from the terminal as

```bash
python -m few.few data_file_name 
```

try ```python -m few.few --help``` to see options.

## Examples

Check out [few_example.py](http://github.com/lacava/few/tree/master/docs/few_example.py) to see how to apply FEW to a regression dataset. 

## Publications

If you use Few, please reference our publications:

La Cava, W., and Moore, J.H. A general feature engineering wrapper for machine learning using epsilon-lexicase survival. *Proceedings of the 20th European Conference on Genetic Programming (EuroGP 2017)*, Amsterdam, Netherlands.
[preprint](http://williamlacava.com/pubs/evostar_few_lacava.pdf)

La Cava, W., and Moore, J.H. Ensemble representation learning: an analysis of fitness and survival for wrapper-based genetic programming methods. *GECCO '17: Proceedings of the 2017 Genetic and Evolutionary Computation Conference*. Berlin, Germany. [arxiv](https://arxiv.org/abs/1703.06934)

## Acknowledgments

This method is being developed to study the genetic causes of human disease in the [Epistasis Lab at UPenn](http://epistasis.org). Work is partially supported by the [Warren Center for Network and Data Science](http://warrencenter.upenn.edu). Thanks to Randy Olson and [TPOT](http://github.com/rhiever/tpot) for Python guidance. 

