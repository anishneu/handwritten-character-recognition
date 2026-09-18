# Handwritten Character Recognition

Bachelor's CS Project (2018-22) - 002

A convolutional neural network that recognizes handwritten English capital
letters (A-Z) from images, trained on the [A-Z Handwritten Alphabets](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
dataset. A pretrained model (**98.2% validation accuracy** across all 26
letters) is included, so you can run predictions right away without
training anything yourself.

## How it works

1. **Preprocessing** — an input image is blurred, converted to grayscale,
   and thresholded to a black-and-white silhouette so it looks like the
   training data (a light stroke on a dark background).
2. **Resizing** — the silhouette is resized to 28x28 pixels, matching the
   dataset's image size.
3. **Classification** — a CNN predicts a probability for each of the 26
   letters, and the highest-probability letter is returned.

### Model architecture

```
Conv2D(32, 3x3) -> MaxPool(2x2)
Conv2D(64, 3x3) -> MaxPool(2x2)
Conv2D(128, 3x3) -> MaxPool(2x2)
Flatten
Dense(64, relu)
Dense(128, relu)
Dense(26, softmax)
```

Trained with the Adam optimizer and categorical cross-entropy loss, with
learning-rate reduction and early stopping on validation loss.

## Project structure

```
.
├── src/
│   ├── model.py       # CNN architecture and A-Z label mapping
│   ├── dataset.py     # CSV loading and preprocessing for training
│   ├── train.py       # Training script (CLI)
│   └── predict.py     # Inference on a single image (CLI)
├── models/
│   └── model_hand.h5  # Pretrained model, ready to use
├── samples/            # Example letter images to try predictions on
├── data/               # Place the training CSV here (see data/README.md)
└── requirements.txt
```

## Setup

Requires Python 3.9+.

```bash
python -m venv venv
venv\Scripts\activate        # on Windows
source venv/bin/activate     # on macOS/Linux
pip install -r requirements.txt
```

## Predicting a letter

Run the pretrained model against one of the sample images:

```bash
python -m src.predict --image samples/letter_Q.png
```

This prints the predicted letter and confidence, and saves an annotated
copy of the image to `outputs/prediction.png`. Add `--show` to also pop up
a window with the result. Try it on your own photo of a handwritten letter
with `--image path/to/your_image.jpg`.

`samples/` has two kinds of examples: `letter_*.png` files pulled straight
from the dataset, and `img_*.jpg` files, which are stylized/decorative
fonts (a graffiti-style "B", a cursive "Q") — a tougher, out-of-distribution
test of how well the model generalizes beyond plain handwriting. The
bundled model gets all of them right.

## Training your own model

1. Download the dataset CSV as described in [`data/README.md`](data/README.md).
2. Run:

   ```bash
   python -m src.train --data data/A_Z_Handwritten_Data.csv --epochs 15
   ```

   Useful flags: `--batch-size`, `--output` (where to save the model).

   Note: the raw dataset CSV is sorted by label (all A's, then all B's, and
   so on). If you use a truncated copy of it, take a stratified sample
   across all 26 letters rather than just the first N rows, or you'll end
   up training on only the first few letters.

The trained model is saved to `models/model_hand.h5` by default, overwriting
the bundled pretrained one.

## License

MIT — see [LICENSE](LICENSE).
