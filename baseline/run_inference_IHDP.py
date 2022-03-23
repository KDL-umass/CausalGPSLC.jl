import toml
import sys
import numpy as np
import pandas as pd
from bartpy.sklearnmodel import SklearnModel

if __name__ == "__main__":
    experiment = sys.argv[1]
    config_path = "../experiments/config/IHDP/" +experiment+ ".toml"
    with open(config_path, "r") as f:
        config = toml.load(f)

    # generate X, T, Y here
    x = np.linspace(0, 5, 3000)
    X = pd.DataFrame(x).sample(frac=1.0).values
    T = np.random.binomial(1, 0.2, 3000).reshape(-1, 1)

    x = np.hstack([X, T])
    y = np.random.normal(0, 0.5, size=3000) + np.sin(X[:, 0]) + np.cos(5 * X[:, 0])

    # BART
    # shape parameter of the prior sigma_a = 0.001
    # shape parameter of the prior sigma_b = 0.001
    # probability of choosing a grow mutation p_grow = 0.5
    # probability of choosing a prune mutation p_prune = 0.5
    # prior parameter on tree structure, alpha = 0.95, beta = 2.0
    model = SklearnModel(n_samples=100,
                         n_burn=50,
                         n_trees=100,
                         store_in_sample_predictions=False)
    model.fit(X, y)
    pred = model.predict(X)
    print(pred)


