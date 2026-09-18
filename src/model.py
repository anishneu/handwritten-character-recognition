"""CNN architecture and label mapping shared by training and inference."""

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 28
NUM_CLASSES = 26

# The A-Z Handwritten Alphabets dataset labels each row 0-25 for A-Z.
LABELS = {i: chr(ord("A") + i) for i in range(NUM_CLASSES)}


def build_model() -> Sequential:
    """Build the CNN used to classify 28x28 grayscale letter images."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu",
               input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        MaxPool2D(pool_size=(2, 2), strides=2),

        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPool2D(pool_size=(2, 2), strides=2),

        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="valid"),
        MaxPool2D(pool_size=(2, 2), strides=2),

        Flatten(),
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
