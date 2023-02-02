import cv2

# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the input image
# img = cv2.imread("IMG_3846.jpg")
img = cv2.imread("input.png")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (127, 255, 0), 2)

# Display the output
cv2.imshow("Faces detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
