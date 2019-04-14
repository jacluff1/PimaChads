import forestci as fci
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
#from treeinterpreter import treeinterpreter as ti

from sklearn.ensemble import RandomForestRegressor


# Data preparation.

def normalize(df, on=None, ignore=(), offset=1e-5):
    on = on if on is not None else df
    for col in set(df.columns) - set(ignore):
        df[col] -= df[col].min()
        df[col] /= df[col].max() - df[col].min() + offset
    return df


if __name__ == "__main__":
    np.random.seed(1)

    target = "saleprice"

    df = {}
    X = {}
    Y = {}

    df["train"] = pd.read_csv("../data_sets/train.csv")
    df["validate"] = pd.read_csv("../data_sets/validate.csv")
    df["test"] = pd.read_csv("../data_sets/test.csv")

    df["train"] = normalize(df["train"], ignore=[target])
    df["validate"] = normalize(df["validate"], df["train"], ignore=[target])
    df["test"] = normalize(df["test"], df["train"], ignore=[target])

    # remove largest
    df["test"] = df["test"].loc[df["test"]["saleprice"] < 3250000]

    for data in df:
        df[data] = df[data].drop(columns="mindisthosp/miles")

        df[data]["lat2"] = df[data]["lat"]**2
        df[data]["lon2"] = df[data]["lon"]**2
        df[data]["lat1lon1"] = df[data]["lat"] * df[data]["lon"]

        df[data]["lat3"] = df[data]["lat"]**3
        df[data]["lon3"] = df[data]["lon"]**3
        df[data]["lat1lon2"] = df[data]["lat"] * df[data]["lon"]**2
        df[data]["lat2lon1"] = df[data]["lat"]**2 * df[data]["lon"]

        df[data]["openness"] = df[data]["sqft"] / (df[data]["rooms"] + 1e-8)

        print(df[data][target].max())

        X[data] = np.array(df[data].drop(columns=target))
        Y[data] = df[data].as_matrix(columns=[target])


    # Grid search.

    r2_best = 0
    model_best = None
    n_estimators_best = None
    max_depth_best = None
    min_samples_leaf_best = None

    for n_estimators in [50]:
    #for n_estimators in [10, 50, 100, 200]:
        for max_depth in [9]:
        #for max_depth in [2, 3, 4, 5, 6, 7, 8, 9]:
            for min_samples_leaf in [5]:
            #for min_samples_leaf in [1, 2, 3, 4, 5, 10]:
                np.random.seed(1)

                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

                model.fit(X["train"], Y["train"])

                r2 = model.score(X["validate"], Y["validate"])

                print(f"RF validation R^2={r2}, n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
                if r2 > r2_best:
                    r2_best = r2
                    model_best = model
                    n_estimators_best = n_estimators
                    max_depth_best = max_depth
                    min_samples_leaf_best = min_samples_leaf

    print(f"best RF validation R^2={r2_best}, n_estimators={n_estimators_best}, max_depth={max_depth_best}, min_samples_leaf_best={min_samples_leaf_best}")

    test_r2 = model_best.score(X["test"], Y["test"])
    print(f"RF test R^2={test_r2}")

    #no geo
    #square RF validation R^2=0.9073309923866705, n_estimators=200, max_depth=9, min_samples_leaf_best=2
    #square RF test R^2=0.8848386935020986
    #cubic RF validation R^2=0.9076529536242397, n_estimators=50, max_depth=9, min_samples_leaf_best=5
    #cubic RF test R^2=0.8840714066246745


    #with geo
    #best RF validation R^2=0.9089340539185148, n_estimators=200, max_depth=5, min_samples_leaf_best=10
    # RF test R^2=0.5511563166786231

    #with openness
    #validation R^2=0.9070573696679025
    #without openness
    #validation R^2=0.906949890999903

    '''
    # Tree interpreter.
    prediction, bias, contributions = ti.predict(model_best, X["test"])
    l = (list(contributions))
    m = [max(row) for row in l]
    print(m)

    features = set(df["train"].columns) - {target}
    for i in range(len(X["test"])):
        print("Instance", i)
        print("Bias (trainset mean)", bias[i])
        print("Feature contributions:")
        for c, feature in sorted(zip(contributions[i],
                                     features),
                                 key=lambda x: -abs(x[0])):
            print(feature, round(c, 2))
        print("-" * 20)
    '''

    # Visualization
    dpi = 300

    Y_hat = model_best.predict(X["test"])
    df_hat = df["test"].copy()
    df_hat[target] = Y["test"]
    df_hat["predicted"] = Y_hat
    df_hat["percent_residual"] = abs(df_hat["predicted"] - df_hat[target]) / df_hat[target]

    df_hat_only = df_hat[[target, "predicted"]]
    df_hat_only.to_csv("y_hat.csv", index=False)

    # Geographic residuals
    df_geographic = pd.DataFrame()
    df_geographic["lat"] = df_hat["lat"]
    df_geographic["lon"] = df_hat["lon"]
    df_geographic["percent_residual"] = abs(df_hat["predicted"] - df_hat[target]) / df_hat[target]
    # Residual color scheme
    cmap = cm.get_cmap("bwr")
    norm = (df_geographic["percent_residual"] - df_geographic["percent_residual"].min()) /\
           (df_geographic["percent_residual"].max() - df_geographic["percent_residual"].min())
    plt.figure()
    #plt.scatter(df_geographic["lon"], df_geographic["lat"], s=.2)
    plt.scatter(df_geographic["lon"], df_geographic["lat"], c=norm, s=.5, cmap=cmap)
    plt.colorbar()
    plt.savefig('Abs_Geo_Residuals.png', dpi=dpi)

    # Predicted vs Actual

    plt.figure()
    # fig, ax = plt.subplots()
    # ax.set_aspect(aspect="equal")
    plt.scatter(Y["test"], Y_hat, s=4)
    plt.plot(range(0, Y["test"].max()), range(0, Y["test"].max()), 'k--')
    plt.gca().set_xticklabels(['${:.2f}M'.format(x / 1e6) for x in plt.gca().get_xticks()])
    plt.xlabel("Predicted Price")
    plt.gca().set_yticklabels(['${:.2f}M'.format(y / 1e6) for y in plt.gca().get_yticks()])
    plt.ylabel("Actual Price")
    plt.savefig('Predicted_vs_Sale_Price.png', dpi=dpi)

    plt.figure()
    plt.scatter(Y["test"], df_hat["percent_residual"], s=4)
    plt.gca().set_xticklabels(['${:.2f}M'.format(x / 1e6) for x in plt.gca().get_xticks()])
    plt.xlabel("Actual Price")
    plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()])
    plt.ylabel("Percent Absolute Error")
    plt.savefig('Percent_Absolute_Error_vs_Sale_Price.png', dpi=dpi)

    plt.figure()
    plt.scatter(Y["test"], df_hat["percent_residual"], s=4)
    plt.xlim(0, 2e6)
    plt.ylim(0, 1)
    plt.title("Percent Absolute Error vs Actual Price Restricted to <100%")
    plt.gca().set_xticklabels(['${:.2f}M'.format(x / 1e6) for x in plt.gca().get_xticks()])
    plt.xlabel("Actual Price")
    plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()])
    plt.ylabel("Percent Absolute Error")
    plt.savefig('Percent_Absolute_Error_vs_Sale_Price_restricted.png', dpi=dpi)

    plt.figure()
    plt.scatter(np.log10(Y["test"]), df_hat["percent_residual"], s=4)
    plt.title("Percent Absolute Error vs Log Sale Price")
    plt.savefig('Percent_Absolute_Error_vs_Log_Sale_Price.png', dpi=dpi)

    # Median absolute percent error by house price.

    # width = 1e5
    # buckets = [ for i in range(0, Y["test"].max(), width)]
    # medians = [np.median(bucket) for bucket in buckets]

    # Median absolute percent error by percentile.
    # q = 10
    # df_quantiles = [df_hat.quantile(q=i/q) for i in range(1, q)]
    # print(df_quantiles)

    # Confidence intervals
    # Confidence interval vs house price

    # variance = fci.random_forest_error(model_best, X["train"], X["test"])
    # plt.figure()
    # plt.errorbar(Y["test"], Y_hat, yerr=np.sqrt(variance), fmt='o')
    # plt.plot(range(0, Y["test"].max()), range(0, int(Y_hat.max())), 'k--')
    # plt.gca().set_xticklabels(['${:.2f}M'.format(x / 1e6) for x in plt.gca().get_xticks()])
    # plt.xlabel("Predicted Price")
    # plt.gca().set_yticklabels(['${:.2f}M'.format(y / 1e6) for y in plt.gca().get_yticks()])
    # plt.ylabel("Actual Price")
    # plt.savefig('Confidence.png', dpi=dpi)
    #


