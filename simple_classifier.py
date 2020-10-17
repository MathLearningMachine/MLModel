import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to image file")
args = vars(ap.parse_args)

model = load_model("classification.model")

img = cv2.imread(args["image"])
img = cv2.resize(img, (32, 32))
img = img_to_array(img)

preds = model.predict(img)

print(preds)