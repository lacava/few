[![Build Status](https://travis-ci.org/lacava/few.svg?branch=master)](https://travis-ci.org/lacava/few)
[![Code Health](https://landscape.io/github/lacava/few/master/landscape.svg?style=flat)](https://landscape.io/github/lacava/few/master)
[![Coverage Status](https://coveralls.io/repos/github/lacava/few/badge.svg?branch=master)](https://coveralls.io/github/lacava/few?branch=master)

Few
===

**Few** is a **Feature Engineering Wrapper** for sci-kitlearn. FEW looks for a set of feature transformations that work best with a specified machine learning algorithm in order to improve model estimation and prediction. In doing so, FEW is able to provide the user with a set of concise, engineered features that describe their data.

Install
===
```pip install few```

Usage
===
Few uses the same nomenclature as [sklearn](http://scikit-learn.org/) supervised learning modules. Given a set of data with variables X and target Y, you can call Few in python as:

```learner = Few(X,Y)```

or specify a machine learning algorithm as:

```learner = Few(X,Y,machine_learner = 'lasso')```

Then optimize the set of feature transformations using the ```fit()``` method:

```learner.fit()```

You have now learned a set of feature tranformations for your data. See the documentation (forthcoming) for more information.

Acknowledgments
===
This method is being developed to study the genetic causes of human disease in the [Epistasis Lab at UPenn](http://epistasis.org). Work is partially supported by the [Warren Center for Network and Data Science](http://warrencenter.upenn.edu). Thanks to Randy Olson and [TPOT](http://github.com/rhiever/tpot) for Python guidance. 

