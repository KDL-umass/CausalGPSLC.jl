import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from bartpy.initializers.initializer import Initializer
from bartpy.sklearnmodel import SklearnModel
from bartpy.diagnostics.trees import plot_tree_depth
from bartpy.diagnostics.features import plot_feature_split_proportions
from bartpy.diagnostics.residuals import plot_qq

from bartpy.samplers.oblivioustrees.treemutation import get_tree_sampler
import statsmodels.api as sm
from bartpy.extensions.baseestimator import ResidualBART
from bartpy.extensions.ols import OLS


def run(alpha, beta, n_trees, size=100):
    import warnings

    warnings.simplefilter("error", UserWarning)
    x = np.linspace(0, 5, size)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=size) + np.sin(x)

    model = OLS(stat_model=sm.OLS,
                n_samples=500,
                n_burn=50,
                n_trees=n_trees,
                alpha=alpha,
                beta=beta,
                n_jobs=1,
                n_chains=1)
    model.fit(X, y)
    plt.plot(y)
    plt.plot(model.predict())
    plt.show()
    plot_tree_depth(model)
    plot_feature_split_proportions(model)
    plot_qq(model)
    #null_distr = null_feature_split_proportions_distribution(model, X, y)
    #print(null_distr)
    return model, x, y


if __name__ == "__main__":
    import cProfile
    from datetime import datetime as dt
    print(dt.now())

    run(0.95, 2., 50)
    #cProfile.run("run(0.95, 2., 50)")
    print(dt.now())
