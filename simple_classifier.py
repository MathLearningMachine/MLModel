import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to image file")
args = vars(ap.parse_args)

model = load_model("classification.model")

img = cv2.imread("exclaim.jpg")
img = cv2.resize(img, (32, 32))
img = img_to_array(img)

preds = model.predict(np.array([img]))

labelNames = ["-", "!", "(", ")", ",", "[", "]", "{", "}", "+", "=", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "alpha", "ascii_124", "b", "beta", "C", "cos", "d", "Delta", "div", "e", "exists", "f", "forall", "forward_slash", "G", "gamma", "geq", "gt", "H", "i", "in", "infty", "int", "j", "k", "l", "lambda", "ldots", "leq", "lim", "log", "lt", "M", "mu", "N", "neq", "o", "p", "phi", "pi", "pm", "prime", "q", "R", "rightarrow", "S", "sigma", "sin", "sqrt", "sum", "T", "tan", "theta", "times", "u", "v", "w", "X", "y", "z"]
labelNamesSorted = sorted(labelNames)

print(preds)
print(labelNamesSorted[preds.argmax(axis=1)[0]])