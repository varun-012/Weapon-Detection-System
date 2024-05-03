# Python Code for Weapon Detection System
# This code demonstrates a simple implementation of a weapon detection system using YOLOv3 and OpenCV for real-time object detection in video streams or webcam feeds.

import cv2  # Importing Computer Vision Module
import numpy as np   # Importing numpy for numerical operations

# The code initializes a deep neural network (DNN) using the YOLOv3 architecture (yolov3_training_2000.weights and yolov3_testing.cfg files) for object detection. 
# It defines the classes for detection, in this case, a single class - "Weapon".
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# It retrieves the output layer names from the neural network and generates random colors for bounding boxes around detected objects.
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# img = cv2.imread("room_ser.jpg")
# img = cv2.resize(img, None, fx=0.4, fy=0.4)

# This function value() prompts the user to input a file name (with extension) or press Enter to start capturing from the webcam. 
# If Enter is pressed without providing a file name, it sets the value to 0, indicating the webcam capture.
def value():
    val = input("Enter file name or press enter to start webcam : \n")
    if val == "":
        val = 0
    return val

# It initializes the video capture using the filename provided by the user or the webcam.
cap = cv2.VideoCapture(value())

# val = cv2.VideoCapture()
# This loop captures frames from the video feed or webcam, checking if the frame was successfully read. 
# If not, it prints an error message and breaks the loop.
while True:
    _, img = cap.read()
    if not _:
        print("Error: Failed to read a frame from the video source.")
        break
    height, width, channels = img.shape
    # width = 512
    # height = 512

# The code inside the loop processes each frame for object detection and drawing bounding boxes.
# Detecting objects using blobFromImage function
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    if len(indexes) == 0:
        print("No weapon detected in frame")
    else:
        print("Weapon detected in frame")
    font = cv2.FONT_HERSHEY_PLAIN
    
# Parsing detected objects and drawing bounding boxes
# NMS (Non-Maximum Suppression) is applied to filter out redundant overlapping boxes.
# Drawing bounding boxes around detected objects
    for i in range(len(boxes)):
        if i in indexes:
            # Get the coordinates, label, and color for each detected object
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            
            # Draw rectangle and label on the image for the detected object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # # Displaying the processed image with bounding boxes and labels
    # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Finally, after the loop ends (by pressing the "Esc" key), the video capture is released, and all OpenCV windows are closed, terminating the program execution.
cap.release()
cv2.destroyAllWindows()

# End of Program