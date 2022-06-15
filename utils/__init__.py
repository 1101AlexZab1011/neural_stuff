import matplotlib.pyplot as plt
from typing import Any
import pickle


def normal_rows(matr):
    matr_n = matr.copy()

    for i, row in enumerate(matr):
        matr_n[i] /= row.mean()

    return matr_n


def read_pkl(path: str) -> Any:
    with open(
        path,
        'rb'
    ) as file:
        content = pickle.load(
            file
        )
    return content


def save_pkl(content: Any, path: str) -> None:
    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')

    pickle.dump(content, open(path, 'wb'))


def plot_data_sample(X, P, Y, i, j):
    sig = X[i, 0, j, :]
    filtered_data = Y[i, 0, j, :]
    predicted_data = P[i, 0, j, :]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(sig)
    ax2.plot(filtered_data)
    ax3.plot(predicted_data)
    fig.set_size_inches(15, 5)
    plt.show()
