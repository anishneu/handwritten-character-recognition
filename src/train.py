"""Train the handwritten character recognition CNN.

Example:
    python -m src.train --data data/A_Z_Handwritten_Data.csv --epochs 10
"""

import argparse
import os

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.dataset import load_dataset
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/A_Z_Handwritten_Data.csv",
        help="Path to the training CSV (see data/README.md for how to get it).",
    )
    parser.add_argument(
        "--output", default="models/model_hand.h5",
        help="Where to save the trained model.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.data} ...")
    train_x, train_y, test_x, test_y = load_dataset(args.data)
    print(f"Train samples: {train_x.shape[0]}, Test samples: {test_x.shape[0]}")

    model = build_model()
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.0001),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ]

    history = model.fit(
        train_x, train_y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(test_x, test_y),
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"Model saved to {args.output}")

    val_acc = history.history["val_accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    print(f"Final validation accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
