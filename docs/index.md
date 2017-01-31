Feature Engineering Wrapper
===
**Few** is a feature engineering wrapper that pairs with any ML estimator to generate a readable representation that facilitates learning. It is implemented to pair easily with any [scikit-learn](http://sklearn.org)-base estimator. 

Few uses genetic programming to generate, search and update engineered features. It incorporates feedback from the ML process to select important features, while also scoring them internally. 

Publications
===
If you use Few, please reference our publication:

La Cava, W., and Moore, J. *A general feature engineering wrapper for machine learning using epsilon-lexicase survival*. Proceedings of the 20th European Conference on Genetic Programming (EuroGP 2017), Amsterdam, Netherlands.

(A preprint is available [here.](http://williamlacava.com/pubs/evostar_few_lacava.pdf)) 


Examples
===
Check out [few_example.py](http://github.com/lacava/few/tree/master/docs/few_example.py) to see how to apply FEW to a regression dataset. 


