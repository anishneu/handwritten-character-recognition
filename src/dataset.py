"""Loading and preprocessing for the A-Z Handwritten Alphabets CSV dataset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.model import IMAGE_SIZE, NUM_CLASSES


def load_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load the dataset CSV and return train/test image and one-hot label arrays.

    The CSV has no header row: the first column is the label (0-25 for A-Z)
    followed by 784 pixel columns (28x28), matching the Kaggle "A-Z
    Handwritten Alphabets in .csv format" dataset.
    """
    data = pd.read_csv(csv_path, header=None).astype("float32")

    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_x = _to_images(train_x)
    test_x = _to_images(test_x)

    train_y = to_categorical(train_y, num_classes=NUM_CLASSES)
    test_y = to_categorical(test_y, num_classes=NUM_CLASSES)

    return train_x, train_y, test_x, test_y


def _to_images(flat_pixels: np.ndarray) -> np.ndarray:
    """Reshape flat pixel rows into normalized (N, 28, 28, 1) images."""
    images = flat_pixels.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    return images / 255.0
