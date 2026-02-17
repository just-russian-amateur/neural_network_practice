import tensorflow as tf
from typing import Tuple

from config import NUM_WORDS


def create_cnn(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """
    Создает и компилирует CNN модель для классификации.

    Args:
        input_shape: форма входных данных (H, W, C)

    Returns:
        model: tf.keras.Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(NUM_WORDS)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    return model
