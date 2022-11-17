import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def train_expanded_model(expanded_queues_file: str, out_folder: str, past_timesteps: int,
                         future_timesteps: int):
    os.makedirs(out_folder, exist_ok=True)

    df = pd.read_csv(expanded_queues_file)
    past_times = range(-past_timesteps + 1, 1)
    future_times = range(1, future_timesteps + 1)
    X_cols = [f"i{k}[{t}]" for k in range(5) for t in past_times] + \
             [f"e[{t}]" for t in past_times]
    Y_cols = [f"e[{t}]" for t in future_times]
    X, Y = df[X_cols].to_numpy(), df[Y_cols].to_numpy()
    X, Y = np.nan_to_num(X), np.nan_to_num(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    pipe = make_pipeline(LinearRegression())
    pipe = pipe.fit(X_train, Y_train)
    score = pipe.score(X_test, Y_test)
    Y_pred = pipe.predict(X_test)
    mse = mean_squared_error(Y_test, np.maximum(np.zeros_like(Y_pred), Y_pred), squared=False)
    y_mean = np.mean(Y_test)
    print(f"Learned model with score {score}, RMSE={mse}, Y_mean={y_mean}")
    model = os.path.join(out_folder, f"expanded-model.pickle")
    with open(model, "wb") as file:
        pickle.dump(pipe, file)
