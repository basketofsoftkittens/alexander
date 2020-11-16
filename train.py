"""

Original Author: Alex Cannan

Modifying Author: You!

Date Imported:

Purpose: This file contains a script meant to train a model.

"""


import os

import sys
import model
import utils
import tensorflow as tf
from tensorflow import keras
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))


DATA_DIR = os.path.join(".", "data")

BIN_DIR = os.path.join(DATA_DIR, "bin")

OUTPUT_DIR = os.path.join(".", "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


np.random.seed(2020)
epoch = 1_000
batch_size = 8
train_split = 0.8
num_train = 13_580
num_test = 4_000
num_valid = 1_000

mos_list = utils.read_list(os.path.join(DATA_DIR, "mos_list.txt"))
train_idx = np.random.randint(0, len(mos_list), int(train_split * len(mos_list)))
mos_list = np.array(mos_list)

train_list = mos_list[train_idx]
valid_list = np.delete(mos_list, train_idx)

train_data = utils.data_generator(
    train_list, BIN_DIR, frame=True, batch_size=batch_size
)
valid_data = utils.data_generator(
    valid_list, BIN_DIR, frame=True, batch_size=batch_size
)

MOSNet = model.CNN()
model = MOSNet.build()

try:
    new_model = tf.keras.models.load_model("output/mos_16.h5")
    model = new_model
    print("Loaded model checkpoint")
except Exception as e:
    print("Failed to load weights", e)


callbacks = [
    keras.callbacks.ModelCheckpoint("output/mos_{epoch}.h5"),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
    ),
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs", write_graph=True, write_images=True, update_freq="batch",
    ),
]


model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3), loss={"avg": "mse", "frame": "mse"},
)


tr_steps = int(num_train / batch_size)
val_steps = int(num_valid / batch_size)

history = model.fit(
    train_data,
    steps_per_epoch=tr_steps,
    epochs=epoch,
    callbacks=callbacks,
    validation_data=valid_data,
    validation_steps=val_steps,
    verbose=1,
)

print(history)
