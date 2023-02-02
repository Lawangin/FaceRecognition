import sys

import cv2

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the video file
video = cv2.VideoCapture("input.mp4")

# Check if the video was successfully loaded
if not video.isOpened():
    print("Error opening video")
    sys.exit()

# Define the colors and labels
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
labels = ['Person 1', 'Person 2', 'Person 3', 'Person 4', 'Person 5', 'Person 6']

# Read frames from the video
while True:
    ret, frame = video.read()

    # Check if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30))

    # Store the boxes and their scores
    boxes = []
    scores = []

    # Draw rectangles around the faces
    for i, (x, y, w, h) in enumerate(faces):
        boxes.append([x, y, x + w, y + h])
        scores.append(1)

    # Perform NMS on the boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)

    # Draw the filtered boxes
    for i in indices:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1])
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i % len(colors)], 2)
        cv2.putText(frame, labels[i % len(labels)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i % len(colors)],
                    2)

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Wait for a key event and exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video
video.release()

# Close all windows
cv2.destroyAllWindows()
