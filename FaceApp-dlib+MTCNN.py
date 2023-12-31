
import dlib
import numpy as np
import cv2
from mtcnn import MTCNN
import warnings
warnings.filterwarnings("ignore")

path1 = "aadhardata/salmankhan.jpg"
path2 = "sampledata/salmansample.jpg"

def getFace(img):
    face_detector = MTCNN()
    face_info = face_detector.detect_faces(img)

    for faces in face_info:
        x, y, width, height = faces['box']

    shape = dlib.rectangle(x, y, x+width, y+height) 
    return shape

def encodeFace(image):
  face_location = getFace(image)
  pose_predictor = dlib.shape_predictor('extra-files/shape_predictor_68_face_landmarks.dat')
  face_landmarks = pose_predictor(image, face_location)
  face_encoder = dlib.face_recognition_model_v1('extra-files/dlib_face_recognition_resnet_model_v1.dat')
  face = dlib.get_face_chip(image, face_landmarks)
  encodings = np.array(face_encoder.compute_face_descriptor(face))
  return encodings

def getSimilarity(image1, image2):
  face1_embeddings = encodeFace(image1)
  face2_embeddings = encodeFace(image2)
  return np.linalg.norm(face1_embeddings-face2_embeddings)

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

distance = getSimilarity(img1,img2)
print(distance)
if distance < .6:
  print("Faces are of the same person.")
else:
  print("Faces are of different people.")
