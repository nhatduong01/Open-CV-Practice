import numpy as np
import cv2 as cv


def rescaleFrame(frame, scale_value=0.5):
    height = int(frame.shape[0]*scale_value)
    width = int(frame.shape[1]*scale_value)
    dimensions = (height, width)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


haar_cascades = cv.CascadeClassifier('Haar_face_detection.xml')
people = ["Thao Nguyen", "Emma Watson",
          "Ariana Grande", "Camila Cabello"]
# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_regconizer = cv.face.LBPHFaceRecognizer_create()
face_regconizer.read('face_trained.yml')

img = cv.imread(
    R'C:\Users\ADMIN\Desktop\Basic of Machine Learning\Open CV\FaceRegconition\Validation\em1.jpg')
#img = rescaleFrame(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)
# detect the face in the face
faces_rect = haar_cascades.detectMultiScale(gray, 1.1, 7)
print(f'The number of person is {len(faces_rect)}')
for (x, y, w, h) in faces_rect:
    faces_crop = gray[x:x+w, y:y+h]
    label, confidence = face_regconizer.predict(faces_crop)
    print(
        f'Lable is  ={people[label]} with a confidence level of {confidence}')
    cv.putText(img, str(people[label]), (20, 20),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y + h), (0, 0, 255), thickness=2)
    cv.imshow("After detection : ", img)
cv.waitKey(0)
