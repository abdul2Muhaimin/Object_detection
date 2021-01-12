# The Spark Foundation GRIP 2k21
# Name: Abdul Muhaimin
# IOT and Computer Vision
# Task 1 (Object Detection)

# importing Libraries
import cv2 # for computer vision
import numpy as np # for numerical computation

# import coco.name
categories = []
with open('coco.names','r') as f:
    categories = f.read().splitlines()

video_cap = cv2.VideoCapture(0)
# if video capture(0) its means the video is capturing with your default web cam, or if (1) it will use the external
# web cam, if we give the path of any video inside this function, the model will detect the objects using this video.

# Importing Weights and Config Files
NN = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
while True:
    _, img = video_cap.read()
    height, width, _ = img.shape # shape is used tto check the size of your video
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # blob from image creates 4 dimensional blob from image, optionally re-sizes and crop image from center,
    # subtract mean values, scales values by scale factor, swapBlue and red channels/
    NN.setInput(blob)
    output_layers_names = NN.getUnconnectedOutLayersNames()
    # it will return the names of layers with unconnected  outputs.
    layerOut = NN.forward(output_layers_names)


    bounding_boxes = []
    accuracy = []
    classIds = []


    for output in layerOut:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bounding_boxes.append([x, y, w, h])
                accuracy.append((float(confidence)))
                classIds.append(classId)
    indexes = cv2.dnn.NMSBoxes(bounding_boxes, accuracy, 0.5, 0.4)
    fonts = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(bounding_boxes), 3))
    

    if len(indexes) != 0:
        for i in indexes.flatten():
            x, y, w, h = bounding_boxes[i]
            label = str(categories[classIds[i]])
            confidence = str(round(accuracy[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (w + h, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), fonts, 2, (0, 0, 0), 2)
            cv2.imshow('image', img)
            key = cv2.waitKey(1)
            if key == 27:
                break

cap.release()
cv2.destroyAllWindows()


