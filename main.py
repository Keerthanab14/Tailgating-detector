import numpy
import argparse
import imutils
import time
import cv2
import os
from stream import TestStream
from imutils import rotate
from imutils.video import FPS
# Parse arguments from command line parameters.
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", type=str, default="yolo",
 help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
 help="minimum confidence threshold for detection")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
 help="threshold for non-maxima suppression")
args = vars(ap.parse_args())
# Load the object classifier label names and associate the classes
# with colours for the bounding box.
labelsFile = os.path.sep.join([args["yolo"], "coco.names"])
labels = open(labelsFile).read().strip().split("\n")
numpy.random.seed(100)
colours = numpy.random.randint(0, 255, size=(len(labels), 3),
 dtype="uint8")
    # Define location of cfg and weight files for our model.
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
     # Load our network model with the given configs and determine names
# of the output layers.
neuralNetwork = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layerNames = neuralNetwork.getLayerNames()
layerNames = [layerNames[i[0] - 1]
 for i in neuralNetwork.getUnconnectedOutLayers()]
# Load the videostream and set dimensions to None.
videoStream = TestStream().start()
(H, W) = (None, None)
# Sleep for 2sec to make sure the stream connection is ok and start
# FPS count.
time.sleep(2.0)
fps = FPS().start()
# Initiate a fullscreen window for our stream.
cv2.namedWindow('Live', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Live", cv2.WND_PROP_FULLSCREEN,
 cv2.WINDOW_FULLSCREEN)
# Loop until interrupted by pressing q.
while True:
 # On every pass we read the latest frame from the video stream.
 frame = videoStream.read()
 # This is only needed in our test case where we had to mount
 # the camera upside down on the wall. 180 degree rotation fixes
 # the issue.
 frame = rotate(frame, 180)
 fps.update()
 # If we are still missing frame dimensions, grab them. Should
 # only happen on first pass. Notice the order here.
 if W is None or H is None:
    (H, W) = frame.shape[:2]
 # Construct a blob from the input frame with our network models
 # settings and perform a forward pass of the neural network,
 # resulting in our bounding box coordinates and associated
 # confidence levels.
 blob = cv2.dnn.blobFromImage(
 frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
 neuralNetwork.setInput(blob)
 layerOutputs = neuralNetwork.forward(layerNames)
 boxes = []
 confidences = []
 classIDs = []
 # Loop through the output layers.
 for output in layerOutputs:
 # Loop through each of the detections on a given output.
    for detection in output:
 # Each detection has a list of classes and related
 # confidence level. We only want the one with the best
 # confidence level, ie the most likely class for this
 # detection.
        confidenceScores = detection[5:]
 classID = numpy.argmax(confidenceScores)
 confidence = confidenceScores[classID]
 # Make sure the best confidence is above our set
 # confidence threshold.
 if confidence > args["confidence"]:
 # Deduce the related bounding box coordinates in
 # the video frame. Scaling is needed here as our
 # camera resolution differs from model resolution.
 # We use top-left coordinates of the bouding box
 # to determine its location.
    boundingBox = detection[0:4] * numpy.array([W, H,
 W, H])
 (boundingBoxCenterX, boundingBoxCenterY,
 width, height) = boundingBox.astype("int")
 x = int(boundingBoxCenterX - (width / 2))
 y = int(boundingBoxCenterY - (height / 2))
 # Add the bounding box coordinates, confidence and
 # classID to the related data structures.
 boxes.append([x, y, int(width), int(height)])
 confidences.append(float(confidence))
 classIDs.append(classID)
 # We use non-maxima suppression to select most relevant found
 # objects.
 indices = cv2.dnn.NMSBoxes(boxes, confidences,
 args["confidence"],
 args["threshold"])
 # Ensure at least one detection exists.
 if len(indices) > 0:
 # Loop through the indices corresponding to relevant
 # bounding boxes.
    for i in indices.flatten():
 # Grab the bounding box coordinates and dimensions.
     (x, y) = (boxes[i][0], boxes[i][1])
 (w, h) = (boxes[i][2], boxes[i][3])
 # Draw the bounding box rectangle and label onto the
 # frame.
 colour = [int(c) for c in colours[classIDs[i]]]
 cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
 text = "{}: {:.4f}".format(labels[classIDs[i]],
 confidences[i])
 cv2.putText(frame, text, (x, y - 5),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
 # Show the image on screen and check if abort key has been
 # pressed.
 cv2.imshow('Live', frame)
 k = cv2.waitKey(10) & 0xff
 if k == ord('q'):
    break
# Stop FPS counter and videostream, destroy the image window and
# print out gained FPS values.
fps.stop()
videoStream.stop()
cv2.destroyAllWindows()
print("FPS: elapsed time: {:.2f}".format(fps.elapsed()))
print("FPS: approx. FPS: {:.2f}".format(fps.fps()))