# Dataset

Training uses the **A-Z Handwritten Alphabets in .csv format** dataset
(28x28 grayscale images of handwritten letters, one per row, label in the
first column):

https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

Download `A_Z Handwritten Data.csv` from Kaggle, rename/place it as:

```
data/A_Z_Handwritten_Data.csv
```

The file is ~700MB and is not tracked in this repository. You only need it
if you want to retrain the model with [`src/train.py`](../src/train.py); a
pretrained model is already provided at [`models/model_hand.h5`](../models/model_hand.h5)
for running predictions.
