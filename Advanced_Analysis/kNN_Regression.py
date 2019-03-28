import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


if __name__ == "__main__":
    df = {}
    X = {}
    Y = {}

    df["train"] = pd.read_csv("train.csv")
    df["validate"] = pd.read_csv("validate.csv")
    df["test"] = pd.read_csv("test.csv")

    target = "SALEPRICE"

    for data in df:
        X[data] = np.array(df[data].drop(columns=target))
        Y[data] = np.array(df[data][target])

    # Grid search on kNN Regression hyperparameters.

    r2_best = 0
    for n in range(20):
        for norm in [1, 2]:
            for weights in ['uniform', 'distance']:
                model = KNeighborsRegressor(n_neighbors=n, p=norm, weights=weights)
                model.fit(X["train"], Y["train"])
                r2 = model.score(X["validate"], Y["validate"])
                print(f"R^2={r2}, n={n}, norm={norm}, weights={weights}")
                if r2 > r2_best:
                    r2_best = r2
                    n_best = n
                    norm_best = norm
                    weights_best = weights

    print(f"Best model: R^2={r2_best}, n={n_best}, norm={norm_best}, weights={weights_best}")
