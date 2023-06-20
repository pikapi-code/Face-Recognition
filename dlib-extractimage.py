import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()

img = cv2.imread("aadhardata/salmankhan.jpg")

bbox = face_detector(img)

print("Number of Faces ==>",len(bbox))

for faces in bbox:
    x, y, width, height = faces.left(), faces.top(), faces.right() - faces.left(), faces.bottom()-faces.top()
    cv2.rectangle(img, (x,y),(x+width,y+height),(255,0,0),2)

cv2.imwrite("image-extracted/dlib_faces.jpg",img)