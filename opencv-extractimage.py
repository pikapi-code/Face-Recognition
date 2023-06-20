import cv2

root = "aadhardata/"
save = "image-extracted/"

img = cv2.imread(root+'salmankhan.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Load the cascade
face_cascade = cv2.CascadeClassifier('extra-files/haarcascade_frontalface_alt2.xml')
  
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  
# Draw rectangle around the faces and crop the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    faces = img[y:y + h, x:x + w]

cv2.imwrite("image-extracted/opencv_faces.jpg",img)