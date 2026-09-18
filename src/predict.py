"""Run the trained model on a single image of a handwritten letter.

Example:
    python -m src.predict --image samples/img_e.jpg --output outputs/prediction.png
"""

import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.model import IMAGE_SIZE, LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to an image containing a single letter.")
    parser.add_argument("--model", default="models/model_hand.h5", help="Path to the trained model.")
    parser.add_argument("--output", default="outputs/prediction.png", help="Where to save the annotated image.")
    parser.add_argument("--show", action="store_true", help="Also open a window to display the result.")
    return parser.parse_args()


def preprocess(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image into the (1, 28, 28, 1) input the model expects."""
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresh, (IMAGE_SIZE, IMAGE_SIZE))
    return resized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0


def predict_letter(model, image: np.ndarray) -> tuple[str, float]:
    """Return the predicted letter and its confidence for a raw BGR image."""
    model_input = preprocess(image)
    probabilities = model.predict(model_input, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return LABELS[index], float(probabilities[index])


def main() -> None:
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = load_model(args.model)
    letter, confidence = predict_letter(model, image)
    print(f"Prediction: {letter} ({confidence:.1%} confidence)")

    annotated = cv2.resize(image, (400, 440))
    cv2.putText(
        annotated, f"Prediction: {letter} ({confidence:.1%})",
        (20, 410), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 30, 255), 2,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to {args.output}")

    if args.show:
        cv2.imshow("Handwritten character recognition", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
