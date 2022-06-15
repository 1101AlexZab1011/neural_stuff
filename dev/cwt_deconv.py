import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import sys
import os
current_dir = os.path.dirname(os.path.abspath('./'))
if not current_dir in sys.path:
    sys.path.append(current_dir)
from utils import normal_rows, SSIMLoss, read_pkl
from sklearn.model_selection import train_test_split
from alltools.machine_learning.designer import ModelDesign
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_predicted_CWT_tc(Y_true, Y_pred, X):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)
    ax1.plot(X)
    ax1.set_title('Original signal')
    mapable = ax2.imshow(
        Y_true,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
    )
    ax2.set_title('CWT')
    plt.colorbar(mapable, ax=ax2)
    mapable = ax3.imshow(
        Y_pred,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
    )
    ax3.set_title('Prediction')
    plt.colorbar(mapable, ax=ax3)
    plt.show()


X, Y = read_pkl('./Source/cwt_data_256.pkl')

X = np.expand_dims(np.expand_dims(X, 1), 3)

for i, (x, y) in enumerate(zip(X, Y)):
    # X[i] = (x - x.min())/(x.max() - x.min())
    Y[i] = (y - y.min()) / (y.max() - y.min())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
del X, Y

# 1 sec, 256 Hz

# model_des = ModelDesign(
#     tf.keras.Input((1, X_train.shape[2], 1)),
#     tf.keras.layers.Conv2DTranspose(10, (127, 10), activation='relu'),
#     tf.keras.layers.Conv2D(1, (1, 10), activation='relu'),
# )

model_des = ModelDesign(
    tf.keras.Input((1, X_train.shape[2], 1)),
    tf.keras.layers.Conv2DTranspose(10, (70, 10), activation='relu',),
    tf.keras.layers.Conv2DTranspose(10, (70, 10), activation='relu'),
    tf.keras.layers.Conv2D(1, (13, 19), activation='relu'),
)

# model_des = ModelDesign(
#     tf.keras.Input((1, X_train.shape[2], 1)),
#     tf.keras.layers.Conv2DTranspose(10, (30, 10), activation='relu',),
#     tf.keras.layers.Conv2DTranspose(10, (30, 10), activation='relu',),
#     tf.keras.layers.Conv2DTranspose(10, (30, 10), activation='relu',),
#     tf.keras.layers.Conv2D(1, (15, 20), activation='relu'),
#     tf.keras.layers.Conv2D(1, (16, 9), activation='relu'),
# )

# model_des()

model = model_des.build()

model.compile(
    optimizer='adam',
    loss=SSIMLoss
)
model.fit(
    X_train,
    np.expand_dims(Y_train, -1),
    epochs=25,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)],
    batch_size=200
)
Y_p = SSIMLoss(
    model(X_test).numpy().astype(float),
    np.expand_dims(Y_test, -1).astype(float)
).numpy()
print(
    'test loss'
    f'{Y_p}'
)


for i in range(5):
    plot_predicted_CWT_tc(normal_rows(Y_test[i]), normal_rows(Y_p[i]), np.squeeze(X_test[i]))
    plt.savefig(f'/home/user/Downloads/Pics/Dumb/network_out_{i}')
