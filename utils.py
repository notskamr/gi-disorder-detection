# Utility functions for ease of use - training, evaluation, and prediction

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os


def compile_and_fit_model(model: tf.keras.Sequential,
                          train_ds: tf.data.Dataset = None,
                          validation_ds: tf.data.Dataset = None,
                          patience: int = 10,
                          save_fig=None,
                          epochs: int = 50, checkpoint_path=None, early_stopping=True,
                          save_weights_only=False, reduce_lr=True) -> tf.keras.callbacks.History:

    if train_ds is None:
        raise ValueError("train_ds must not be None")
    if validation_ds is None:
        raise ValueError("validation_ds must not be None")

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=patience,
        mode="auto",
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=patience // 2,
        min_lr=0.0001
    )

    callbacks = []
    if early_stopping == True:
        callbacks.append(early_stopping)
    if reduce_lr == True:
        callbacks.append(reduce_lr)
    if checkpoint_path != None:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        monitor="val_accuracy", mode="max",
                                                        save_best_only=True, save_weights_only=save_weights_only, verbose=1)
        callbacks.append(checkpoint)
    # Fit the model
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Plot the performance
    performance_df = pd.DataFrame(history.history)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for ax, metric in zip(axes.flat, ["accuracy", "loss"]):
        performance_df.filter(like=metric).plot(ax=ax)
        ax.set_title(metric.title(), size=15, pad=20)
    plt.show()

    return history


def save_figure_from_history(history: tf.keras.callbacks.History, path: str):
    if 'history' not in history:
        history = history
    else:
        history = history.history
    performance_df = pd.DataFrame(history)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for ax, metric in zip(axes.flat, ["accuracy", "loss"]):
        performance_df.filter(like=metric).plot(ax=ax)
        ax.set_title(metric.title(), size=15, pad=20)
    plt.savefig(path, bbox_inches="tight")


def plot_predictions(model: tf.keras.Sequential,
                     train_ds: tf.data.Dataset = None,
                     class_details: Dict[int, str] = None) -> None:
    plt.figure(figsize=(14, 14))
    for images, labels in train_ds.take(2):
        labels = labels.numpy()
        predicted_labels = np.argmax(model.predict(images), axis=1)
        for i, (actual, pred) in enumerate(zip(predicted_labels, labels)):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if actual == pred:
                plt.title(class_details[labels[i]], color="green", size=9)
            else:
                plt.title(f"{class_details[predicted_labels[i]]}\n"
                          + f"(Actual: {class_details[labels[i]]})",
                          color="red", size=9)
            plt.axis("off")
    plt.show()


def get_predictions(model: tf.keras.Sequential,
                    ds: tf.data.Dataset = None) -> Tuple[np.ndarray, np.ndarray]:
    if ds is None:
        raise ValueError("ds must not be None")

    # Initialize lists to store predicted labels and actual labels
    all_predictions = []
    all_labels = []

    # Disable progress bar
    verbose_backup = tf.keras.utils.get_custom_objects().get('progress_bar')
    tf.keras.utils.get_custom_objects()['progress_bar'] = lambda x: None

    # Predict and store predicted labels and actual labels
    for images, labels in ds:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_labels)
        all_labels.extend(labels.numpy())

    # Restore progress bar
    tf.keras.utils.get_custom_objects()['progress_bar'] = verbose_backup

    return np.array(all_labels), np.array(all_predictions)
