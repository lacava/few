"""
Copyright 2016 William La Cava

This file is part of the FEW library.

The FEW library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The FEW library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the FEW library. If not, see http://www.gnu.org/licenses/.

"""
# test FEW methods
from few.few import FEW
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLarsCV
import pandas as pd
import numpy as np

def test_few_fit_shapes():
    """test_few.py: fit and predict return correct shapes """
    np.random.seed(202)
    # load example data
    boston = load_boston()
    d = pd.DataFrame(data=boston.data)
    print("feature shape:",boston.data.shape)

    learner = FEW(generations=1, population_size=5,
                mutation_rate=0.2, crossover_rate=0.8,
                ml = LassoLarsCV(), min_depth = 1, max_depth = 3,
                sel = 'epsilon_lexicase', tourn_size = 2, random_state=0, verbosity=2,
                disable_update_check=False, fit_choice = 'mse')

    score = learner.fit(boston.data[:300], boston.target[:300])
    print("learner:",learner._best_estimator)
    yhat_test = learner.predict(boston.data[300:])
    test_score = learner.score(boston.data[300:],boston.target[300:])
    print("train score:",score,"test score:",test_score,
    "test r2:",r2_score(boston.target[300:],yhat_test))
    assert yhat_test.shape == boston.target[300:].shape


def test_few_at_least_as_good_as_default():
    """test_few.py: few performs at least as well as the default ML """
    np.random.seed(1006987)
    boston = load_boston()
    d = np.column_stack((boston.data,boston.target))
    np.random.shuffle(d)
    features = d[:,0:-1]
    target = d[:,-1]

    print("feature shape:",boston.data.shape)

    learner = FEW(generations=1, population_size=5,
                mutation_rate=1, crossover_rate=1,
                ml = LassoLarsCV(), min_depth = 1, max_depth = 3,
                sel = 'epsilon_lexicase', fit_choice = 'r2',tourn_size = 2, random_state=0, verbosity=1,
                disable_update_check=False)

    learner.fit(features[:300], target[:300])
    few_score = learner.score(features[:300], target[:300])
    test_score = learner.score(features[300:],target[300:])

    lasso = LassoLarsCV()
    lasso.fit(learner._training_features,learner._training_labels)
    lasso_score = lasso.score(features[:300], target[:300])
    print("few score:",few_score,"lasso score:",lasso_score)
    print("few test score:",test_score,"lasso test score:",lasso.score(features[300:],target[300:]))
    assert few_score >= lasso_score

    print("lasso coefficients:",lasso.coef_)

    # assert False
