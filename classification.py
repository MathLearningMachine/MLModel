from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

def make_cnn(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    # define first layer, CONV => RELU layer
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))

	# softmax classifier
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))

	# return network architecture
    return model

def preprocessImage(image, width, height):
    scaled_image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    array_img = img_to_array(scaled_image)
    return array_img


def load_images(imagePaths, width, height):
    data = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2] # Use directory name as label
        preprocessedImage = preprocessImage(image, width, height)
        data.append(preprocessedImage)
        labels.append(label)
        print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
    
    return (np.array(data), np.array(labels))

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

(data, labels) = load_images(imagePaths, 32, 32)

# scale intensity of pixel channel data
data = data.astype("float") / 255.0

# split train and test sets dynamically
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# 'one-hot' encode labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = make_cnn(width=32, height=32, depth=3, classes=82)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=100, verbose=1)

model.save('classification.model')

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["-", "!", "(", ")", ",", "[", "]", "{", "}", "+", "=", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "alpha", "ascii_124", "b", "beta", "C", "cos", "d", "Delta", "div", "e", "exists", "f", "forall", "forward_slash", "G", "gamma", "geq", "gt", "H", "i", "in", "infty", "int", "j", "k", "l", "lambda", "ldots", "leq", "lim", "log", "lt", "M", "mu", "N", "neq", "o", "p", "phi", "pi", "pm", "prime", "q", "R", "rightarrow", "S", "sigma", "sin", "sqrt", "sum", "T", "tan", "theta", "times", "u", "v", "w", "X", "y", "z"]))

# print out a report
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('report-graph.png')