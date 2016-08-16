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
import pandas as pd
def test_few_fit_shapes():
    """ few.fit and few.predict correct shapes """
    # load example data
    boston = load_boston()
    d = pd.DataFrame(data=boston.data)
    print("feature shape:",boston.data.shape)

    learner = FEW(generations=1000, population_size=1000,
                mutation_rate=0.2, crossover_rate=0.8,
                machine_learner = 'lasso', min_depth = 1, max_depth = 3,
                sel = 'tournament', tourn_size = 2, random_state=0, verbosity=1,
                disable_update_check=False)

    score = learner.fit(boston.data[:300], boston.target[:300])
    print("learner:",learner._best_estimator)
    yhat_test = learner.predict(boston.data[300:])
    test_score = learner.score(boston.data[300:],boston.target[300:])
    print("train score:",score,"test score:",test_score,
    "test r2:",r2_score(boston.target[300:],yhat_test))
    assert False

def test_few_at_least_as_good_as_default():
    """ few performs at least as well as the default ML """
