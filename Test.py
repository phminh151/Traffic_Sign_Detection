import numpy as np
import cv2
import pickle
from helpers.yolo import load_image, load_yolo, detect_objects, get_box_dimensions, draw_labels
import keras
# Yolo model
net, classes, colors, output_layers = load_yolo()
# Load Image
img, height, width, channels = load_image('./images/test.jpg')
# Detect objects
blob, outputs = detect_objects(img, net, output_layers)
# Find boxes
boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
# Load Classifier
classifier = keras.models.load_model('model/traffic_classifier3.h5')
# Draw Labels
indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
	if i in indexes:
		x, y, w, h = boxes[i]
		color = colors[i]
		cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
		crop_img = img[y:y+h, x:x+w]
		label = classifier.predict(crop_img) 
		cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
cv2.imshow("Image", img)
cv2.waitKey()