from mtcnn import MTCNN
import cv2

face_detector = MTCNN()

img = cv2.cvtColor(cv2.imread("aadhardata/salmankhan.jpg"), cv2.COLOR_BGR2RGB)

face_info = face_detector.detect_faces(img)

print("Number of Faces ==>",len(face_info))

for faces in face_info:
    x, y, width, height = faces['box']
    cv2.rectangle(img, (x,y),(x+width,y+height),(255,0,0),2)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite("image-extracted/mtcnn_faces.jpg",img)