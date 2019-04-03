import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.models import Sequential
from sklearn import metrics


def normalize(df, on=None, ignore=(), offset=1e-5):
    on = on if on is not None else df
    for col in set(df.columns) - set(ignore):
        df[col] -= df[col].min()
        df[col] /= df[col].max() - df[col].min() + offset
        df[col] = 2 * df[col] - 1
    return df


def FFNN(X_train, Y_train, layer_widths, epochs):

    D = X_train.shape[1]

    model = Sequential()

    # Input layer.
    model.add(Dense(units=layer_widths[0], activation='tanh', input_shape=(D,)))

    model.add(Dense(units=layer_widths[1]))
    model.add(LeakyReLU(alpha=0.3))

    # Hidden layers.
    for layer_width in layer_widths[2:]:
        model.add(Dense(units=layer_width, activation='tanh'))

    # Regression output layer.
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse']
                  )

    model.fit(X_train, Y_train, epochs=epochs, batch_size=32)

    return model


if __name__ == "__main__":
    np.random.seed(1)

    df = {}
    X = {}
    Y = {}

    df["train"] = pd.read_csv("../data_sets/train.csv")
    df["validate"] = pd.read_csv("../data_sets/validate.csv")
    df["test"] = pd.read_csv("../data_sets/test.csv")

    target = "saleprice"

    df["train"] = normalize(df["train"], ignore=target)
    df["validate"] = normalize(df["validate"], df["train"], ignore=target)
    df["test"] = normalize(df["test"], df["train"], ignore=target)

    for data in df:
        X[data] = np.array(df[data].drop(columns=target))
        Y[data] = np.array(df[data][target])

    # Grid search on kNN Regression hyperparameters.

    r2_best = 0
    model_best = None

    for layer_widths in (
        [100, 100, 50, 20],
        [200, 400, 100, 40],
        [100, 20, 20, 20, 20],
        [100, 100, 20, 20, 20],
        [200, 100, 20, 20, 20],
    ):
        for epochs in [50]:
            # Create the model.
            model = FFNN(X["train"], Y["train"], layer_widths, epochs=epochs)

            # Evaluate the model.
            Y_hat = model.predict(X["validate"])
            r2 = metrics.r2_score(y_true=Y["validate"], y_pred=Y_hat)

            print(f"R^2={r2}, layer_widths={layer_widths}")
            if r2 > r2_best:
                r2_best = r2
                layer_widths_best = layer_widths
                model_best = model

    print(f"Best model: R^2={r2_best}, layer_widths={layer_widths_best}, epochs={epochs}")
    print(model_best.summary())
