# YOLO object detection
import cv2 as cv
import numpy as np
import time

PERSON_ID = 0 # Related to coco.names file (ID)

img = cv.imread('test_image2.jpg')

# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(777)
color = [int(c) for c in np.random.randint(0, 255, (1, 3), dtype='uint8')[0]]

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

net.setInput(blob)
outputs = net.forward(ln)

boxes = []
confidences = []
h, w = img.shape[:2]

####

for channels in outputs:
    for image in channels:
        if np.argmax(image[5:]) == 0 and image[5:][PERSON_ID] > 0.5:
            box = image[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(image[5:][np.argmax(image[5:])]))

indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

count = len(indices)

if count > 0:
    for indice in indices.flatten():
        (x, y, w, h) = boxes[indice]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # Put text count
    text = "Person: {}".format(count)
    cv.putText(img, text, (int(w), int(h)), 0, 1, color, 2)


cv.imshow('window', img)
cv.waitKey(0)