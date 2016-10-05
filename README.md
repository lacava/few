[![Build Status](https://travis-ci.org/lacava/few.svg?branch=master)](https://travis-ci.org/lacava/few)
[![Code Health](https://landscape.io/github/lacava/few/master/landscape.svg?style=flat)](https://landscape.io/github/lacava/few/master)
[![Coverage Status](https://coveralls.io/repos/github/lacava/few/badge.svg?branch=master)](https://coveralls.io/github/lacava/few?branch=master)

Few
===

**Few** is a **Feature Engineering Wrapper** for sci-kitlearn. Few looks for a set of feature transformations that work best with a specified machine learning algorithm in order to improve model estimation and prediction. In doing so, FEW is able to provide the user with a set of concise, engineered features that describe their data.

Install
===
```pip install few```

Usage
===
In a python script, import FEW:

```python
from few.few import FEW
```

Few uses the same nomenclature as [sklearn](http://scikit-learn.org/) supervised learning modules. You can initialize a few learner in python as:

```python
learner = FEW()
```

or specify the generations, population size and machine learning algorithm as:

```python
learner = FEW(generations = 100, population_size = 25, machine_learner = LassoLarsCV())
```

Given a set of data with variables X and target Y, optimize the set of feature transformations using the ```fit()``` method:

```python
learner.fit(X,Y)
```

You have now learned a set of feature tranformations for your data, as well as a predictor that uses the chosen machine learning algorithm with these feaures. Predict your model's response on a new set of variables as

```python
y_pred = learner.predict(X_unseen)
```

You can use the ```transform()``` method to just perform a feature tranformation using the learned features:

```python
X_tranformed = learner.transform(X)
``` 

Call Few from the terminal as

```bash
python -m few.few data_file_name 
```

try ```python -m few.few --help``` to see options.

See the documentation (forthcoming) for more information.

Acknowledgments
===
This method is being developed to study the genetic causes of human disease in the [Epistasis Lab at UPenn](http://epistasis.org). Work is partially supported by the [Warren Center for Network and Data Science](http://warrencenter.upenn.edu). Thanks to Randy Olson and [TPOT](http://github.com/rhiever/tpot) for Python guidance. 

