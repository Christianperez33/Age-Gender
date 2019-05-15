# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
from utils import rgb2gray
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

argparser = argparse.ArgumentParser()

argparser.add_argument("-i", "--image",
	default='./predict/female1.jpg',
	help="path to input image")
argparser.add_argument("-m", "--model",
	default='./Age-Gender_model.h5',
	help="path to trained model file")
argparser.add_argument("-w", "--weights",
	default='./final_weights.hdf5',
	help="path to trained weights file")

args = argparser.parse_args()



image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2GRAY)
output =cv2.imread(args.image)
if image is None:
    print("Could not read input image")
    exit()


image = cv2.resize(image,(32,32))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model(args.model)
model.load_weights(args.weights)
confidence = model.predict(image)


classes = ["Mujer","Hombre"]    
gender = np.argmax(confidence[0])
age = np.argmax(confidence[1])
genLab = "{}: {:.2f}%".format(classes[gender], confidence[0][0][gender] * 100)
ageLab = " Edad:{} ".format(age)
cv2.putText(output, genLab, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(output, ageLab, (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("Age-Gender classification", output)
cv2.waitKey()
cv2.destroyAllWindows()
