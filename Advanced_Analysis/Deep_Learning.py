import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn import metrics


def ANN(X_train, Y_train, layer_widths):

    N = X_train.shape[0]

    model = Sequential()

    # Input layer.
    model.add(Dense(units=layer_widths[0], activation='tanh', input_dim=N))

    # Hidden layers.
    for layer_width in layer_widths[1:]:
        model.add(Dense(units=layer_width, activation='softmax'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse']
                  )

    model.fit(X_train, Y_train, epochs=5, batch_size=32)

    return model


if __name__ == "__main__":
    np.random.seed(1)

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
    for layer_widths in (
        [10, 10],
        [20, 20],
        [100, 10],
        [100, 50],
        [100, 100],
        [50, 50, 20],
    ):
        # Create the model.
        model = ANN(X["train"], Y["train"], layer_widths)
        model.fit(X["train"], Y["train"])

        # Evaluate the model.
        Y_hat = model.predict(X["validate"])
        r2 = metrics.r2_score(y_true=Y["validate"], y_pred=Y_hat)

        print(f"R^2={r2}, layer_widths={layer_widths}")
        if r2 > r2_best:
            r2_best = r2
            layer_widths_best = layer_widths

    print(f"Best model: R^2={r2_best}, layer_widths={layer_widths_best}")
